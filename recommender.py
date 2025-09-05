# recommender.py
import psycopg2
from typing import List, Dict, Tuple, Optional
import math

def interleave_pairs(a, b, region_norm, threshold=0.2, top_n=15):
    """
    Strict interleaving of region (a) and global (b).
    - Include region only if region_norm >= threshold.
    - Alternate picks: region → global → region → global …
    - If region runs out (or all remaining fall below threshold), fill with globals.
    """
    out, seen = [], set()
    ia, ib = 0, 0
    turn = "region"  # start with region

    while len(out) < top_n and (ia < len(a) or ib < len(b)):
        if turn == "region" and ia < len(a):
            iid, score = a[ia]
            ia += 1
            if iid not in seen and region_norm.get(iid, 0.0) >= threshold:
                out.append((iid, score, "region"))
                seen.add(iid)
            turn = "global"  # switch turn regardless

        elif turn == "global" and ib < len(b):
            iid, score = b[ib]
            ib += 1
            if iid not in seen:
                out.append((iid, score, "global"))
                seen.add(iid)
            turn = "region"

        else:
            # If one side is exhausted, always take from the other
            if ia >= len(a) or all(region_norm.get(x[0], 0.0) < threshold for x in a[ia:]):
                # backfill with globals
                while len(out) < top_n and ib < len(b):
                    iid, score = b[ib]
                    ib += 1
                    if iid not in seen:
                        out.append((iid, score, "global"))
                        seen.add(iid)
                break
            else:
                turn = "region"

    return out



class HybridRecommender:
    """
    Hybrid recommender backed by Postgres.

    Tables expected:
      - product_map(item_id INT PRIMARY KEY, description TEXT, is_top100 BOOLEAN)
      - item_popularity(item_id INT, region TEXT, count INT, PRIMARY KEY(item_id, region))
      - rules(
            id SERIAL PRIMARY KEY,
            antecedent_arr INT[],
            consequent_arr INT[],   -- single-item consequents preferred
            confidence FLOAT,
            lift FLOAT,
            score FLOAT
        )

    Key ideas:
      * Cold-start: interleave region + global popular items from item_popularity.
      * Warm-start: rules where antecedent_arr <@ basket (subset).
      * Backfill: if rules < top_n, fill with popularity (excluding basket & already picked).
      * Always map IDs -> product names using product_map.
    """

    def __init__(
        self,
        conn_params: Dict,
        region: str = "GLOBAL",
        rule_table: str = "rules",
        popularity_table: str = "item_popularity",
        product_map_table: str = "product_map",
        antecedent_col: str = "antecedent_arr",
        consequent_col: str = "consequent_arr",
    ):
        self.conn_params = conn_params
        self.region = region
        self.rule_table = rule_table
        self.popularity_table = popularity_table
        self.product_map_table = product_map_table
        self.antecedent_col = antecedent_col
        self.consequent_col = consequent_col

        self.basket: set[int] = set()
        self.id_to_item: Dict[int, str] = {}
        self._load_product_map()

    # ---------------------------
    # Setup / utilities
    # ---------------------------
    def _connect(self):
        return psycopg2.connect(**self.conn_params)

    def _load_product_map(self):
        """Load id->description into memory once (fast lookup)."""
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(f"SELECT item_id, description FROM {self.product_map_table}")
        self.id_to_item = {row[0]: row[1] for row in cur.fetchall()}
        cur.close(); conn.close()

    def set_region(self, region: str):
        self.region = region

    def reset_basket(self):
        self.basket.clear()

    def add_item(self, item: int, top_n: int = 15):
        """Add an item_id to basket and return fresh recs."""
        self.basket.add(int(item))
        return self.recommend(top_n=top_n)

    def remove_item(self, item: int, top_n: int = 15):
        self.basket.discard(int(item))
        return self.recommend(top_n=top_n)

    # ---------------------------
    # 1) Popularity (cold start + backfill)
    # ---------------------------
    def _fetch_popularity(
        self,
        top_n_region: int = 20,
        top_n_global: int = 20,
        exclude_ids: Optional[List[int]] = None,
    ) -> List[Tuple[int, int, str]]:
        """
        Return interleaved popularity list: [(item_id, count, 'region'|'global'), ...]
        Excludes any IDs in exclude_ids (e.g. basket or already-picked).
        """
        exclude_ids = exclude_ids or []
        conn = self._connect()
        cur = conn.cursor()

        # Region-specific (skip basket/already-picked in SQL)
        cur.execute(
            f"""
            SELECT item_id, count
            FROM {self.popularity_table}
            WHERE region = %s
              AND NOT (item_id = ANY(%s))  -- exclude
            ORDER BY count DESC
            LIMIT %s
            """,
            (self.region, exclude_ids, top_n_region * 2),  # fetch extra in case of exclusions
        )
        region_rows = cur.fetchall()

        # Global
        cur.execute(
            f"""
            SELECT item_id, count
            FROM {self.popularity_table}
            WHERE region = 'GLOBAL'
              AND NOT (item_id = ANY(%s))
            ORDER BY count DESC
            LIMIT %s
            """,
            (exclude_ids, top_n_global * 2),
        )
        global_rows = cur.fetchall()

        cur.close(); conn.close()

        # Tag but do NOT interleave here — just return raw rows
        region_tagged = [(iid, score, "region") for iid, score in region_rows]
        global_tagged = [(iid, score, "global") for iid, score in global_rows]
        return region_tagged + global_tagged


    # ---------------------------
    # 2) Rules (warm start)
    # ---------------------------   

    def _fetch_rules(self, basket_ids, fetch_limit=100):
        if not basket_ids:
                return {}
            
        """Fetch candidate consequents based on association rules (antecedent_key/consequent_key are TEXT)."""
        conn = psycopg2.connect(**self.conn_params)
        cur = conn.cursor()
    
        # Convert basket IDs into text array for Postgres
        basket_strs = [str(b) for b in basket_ids]
    
        cur.execute("""
            SELECT consequent_key, confidence, lift, score
            FROM rules
            WHERE string_to_array(antecedent_key, ',') <@ %s::text[]
            ORDER BY score DESC
            LIMIT %s
        """, (basket_strs, fetch_limit))
    
        rows = cur.fetchall()
        cur.close(); conn.close()
    
        basket_set = set(basket_ids)
        recs = {}
        for cons_str, conf, lift, score in rows:
            cons_items = set(map(int, cons_str.split(",")))  # split text "96" → {96}
            # skip if consequent overlaps basket
            if cons_items & basket_set:
                continue
            c = list(cons_items)[0]   # single-item consequent
            recs[c] = max(recs.get(c, 0), score)
        return recs


    # ---------------------------
    # Add helpers inside HybridRecommender
    # ---------------------------
    def _adaptive_alpha(self, basket_size: int, c: float = 3.0) -> float:
        """
        Adaptive alpha in [0,1]. c controls how fast alpha rises with basket size.
        - c ~ 3 is a reasonable start (1 item => 0.25, 3 items => 0.5, 9 items => 0.75).
        """
        if basket_size <= 0:
            return 0.0
        return float(basket_size) / (basket_size + c)
    
    def _normalize_rule_scores(self, rules_dict: Dict[int, float]) -> Dict[int, float]:
        """Min-max normalize rule scores into [0,1]."""
        if not rules_dict:
            return {}
        vals = list(rules_dict.values())
        mn, mx = min(vals), max(vals)
        if mx == mn:
            # single non-zero value => set to 1.0, zeros stay 0.0
            return {iid: (1.0 if v > 0 else 0.0) for iid, v in rules_dict.items()}
        return {iid: (v - mn) / (mx - mn) for iid, v in rules_dict.items()}
    
    def _normalize_pop_scores(self, pop_scores: Dict[int, float]) -> Dict[int, float]:
        """
        Log-transform (log1p) then min-max normalize popularity counts into [0,1].
        pop_scores: {item_id: count}
        """
        if not pop_scores:
            return {}
        ids, vals = zip(*pop_scores.items())
        vals_log = [math.log1p(v) for v in vals]
        mn, mx = min(vals_log), max(vals_log)
        if mx == mn:
            return {iid: (1.0 if v > 0 else 0.0) for iid, v in pop_scores.items()}
        normalized = {}
        for iid, v in pop_scores.items():
            y = math.log1p(v)
            normalized[iid] = (y - mn) / (mx - mn)
        return normalized

    # ---------------------------
    # 3) Main recommend()
    # ---------------------------
    def recommend(self, top_n: int = 15) -> List[Dict]:
        """
        Return a list of dicts with consistent scoring:
          - Cold start: region_score or global_score (normalized separately)
          - Warm start: final_score + rule_norm/pop_norm (normalized on union)
        """
        exclude = list(self.basket)  # never recommend items already in basket
    
        # -------------------------
        # Case A: Cold start
        # -------------------------
        if not self.basket:
            # fetch region + global popularity
            pop_all = self._fetch_popularity(
                top_n_region=top_n,
                top_n_global=top_n,
                exclude_ids=exclude
            )
        
            # Separate sets
            pop_region = [(iid, score) for (iid, score, src) in pop_all if src == "region"]
            pop_global = [(iid, score) for (iid, score, src) in pop_all if src == "global"]
        
            # Normalization per set
            def minmax_norm(vals):
                if not vals:
                    return {}
                min_v, max_v = min(v for _, v in vals), max(v for _, v in vals)
                if max_v == min_v:
                    return {iid: 1.0 for iid, _ in vals}
                return {iid: (v - min_v) / (max_v - min_v) for iid, v in vals}
        
            region_norm = minmax_norm(pop_region)
            global_norm = minmax_norm(pop_global)
        
            # Threshold-aware interleaving
            pop = interleave_pairs(pop_region, pop_global, region_norm, threshold=0.2, top_n=top_n)
    
        
            results = []
            for (iid, score, src) in pop[:top_n]:
                results.append({
                    "item_id": iid,
                    "description": self.id_to_item.get(iid, f"<unknown:{iid}>"),
                    "region_score": region_norm.get(iid) if src == "region" else None,
                    "global_score": global_norm.get(iid) if src == "global" else None,
                    "source": src
                })
            return results

    
        # -------------------------
        # Case B: Warm start
        # -------------------------
        rules_dict = self._fetch_rules(list(self.basket), fetch_limit=top_n * 8)
        ranked_rules = sorted(rules_dict.items(), key=lambda x: x[1], reverse=True)
    
        picked_ids, results = set(), []
        candidate_scores = {}
    
        # Collect rules scores
        for iid, score in ranked_rules:
            if iid not in self.basket:
                candidate_scores[iid] = {"rule": score}
    
        # Collect popularity scores for backfill
        exclude_more = list(self.basket.union(set(candidate_scores.keys())))
        pop = self._fetch_popularity(
            top_n_region=top_n,
            top_n_global=top_n,
            exclude_ids=exclude_more
        )
        for iid, score, src in pop:
            if iid not in self.basket:
                if iid not in candidate_scores:
                    candidate_scores[iid] = {}
                candidate_scores[iid]["pop"] = score
    
        # -------------------------
        # Normalization (on union set)
        # -------------------------
        def minmax_dict(d: Dict[int, float]):
            if not d: return {}
            vals = list(d.values())
            min_v, max_v = min(vals), max(vals)
            return {k: (v - min_v) / (max_v - min_v + 1e-9) for k, v in d.items()}
    
        rule_scores = {iid: s["rule"] for iid, s in candidate_scores.items() if "rule" in s}
        pop_scores = {iid: s["pop"] for iid, s in candidate_scores.items() if "pop" in s}
    
        rule_norm = minmax_dict(rule_scores)
        pop_norm = minmax_dict(pop_scores)
    
        # Adaptive alpha
        c=3
        alpha = min(1.0, max(0.0, len(self.basket) / (len(self.basket)+c)))

    
        # Final scoring
        for iid, comp in candidate_scores.items():
            rn = rule_norm.get(iid, 0.0)
            pn = pop_norm.get(iid, 0.0)
            final_score = alpha * rn + (1 - alpha) * pn
    
            results.append({
                "item_id": iid,
                "description": self.id_to_item.get(iid, f"<unknown:{iid}>"),
                "final_score": final_score,
                "rule_norm": rn if "rule" in comp else None,
                "pop_norm": pn if "pop" in comp else None,
                "source": "rules" if "rule" in comp else "popularity"
            })
    
        # Sort by final_score
        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        return results[:top_n]
