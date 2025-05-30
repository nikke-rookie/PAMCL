from numba import jit
import heapq

@jit(nopython=True)
def find_k_largest(K: int, candidates: list):
    """
    从候选集中找到前K个最大值

    Args:
        K (int): top K
        candidates (list): 评分候选列表 [(item_id, score)]
    
    Returns:
        (ids, k_largest_scores) (list, list): 最大值索引列表, 最大值列表
    """
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates)
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores
