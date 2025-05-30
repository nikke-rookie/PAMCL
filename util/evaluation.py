import math


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        """
        计算推荐系统的命中数(hits): 推荐列表中有多少项在真实的用户行为数据中也被记录过
        
        Args:
            origin (dict): 原始用户行为数据(测试集)
            res (dict): 推荐系统的推荐结果
        
        Returns:
            hit_count (dict): 每个用户的命中数
        """
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        计算命中率: 在测试集中检索到的交互次数 / 测试集中的所有交互次数

        Args:
            origin (dict): 原始数据
            hits (dict): 每个用户命中数
    
        Returns:
            hit_ratio: 返回测试集中检索到的交互次数占所有交互次数的比例，保留五位小数
        """
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return round(hit_num/total_num, 5)

    @staticmethod
    def precision(hits, N):
        """
        计算精确度: 命中数总和/用户数量*top-N

        Args:
            hits (dict): 每个用户的命中数
            N (int): top-N

        Returns:
            float: 推荐系统的精确度，保留五位小数。
        """
        prec = sum([hits[user] for user in hits])
        return round(prec / (len(hits) * N), 5)

    @staticmethod
    def recall(hits, origin):
        """
        计算平均召回率

        Args:
            hits (dict): 每个用户的命中数
            origin (dict): 测试集

        Returns:
            float: 平均召回率
        """
        recall_list = [hits[user]/len(origin[user]) for user in hits]
        recall = round(sum(recall_list) / len(recall_list), 5)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return round(2 * prec * recall / (prec + recall),5)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error+=abs(entry[2]-entry[3])
            count+=1
        if count==0:
            return error
        return round(error/count,5)

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3])**2
            count += 1
        if count==0:
            return error
        return round(math.sqrt(error/count),5)

    @staticmethod
    def NDCG(origin,res,N):
        """
        计算归一化折损累积增益(NDCG)

        Args:
            origin (dict): 测试集
            res (dict): 推荐结果
            N (int): top-N

        Returns:
            float: 平均NDCG值
        """
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0/math.log(n+2, 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0/math.log(n+2, 2)
            sum_NDCG += DCG / IDCG
        return round(sum_NDCG/len(res), 5)

def ranking_evaluation(origin, res, N):
    """
    通过计算top-N的各种评估指标来评估排名结果的质量

    Args:
        origin (dict): 真实结果(测试集)
        res (dict): 模型预测的结果集
        N (list): top-N 值

    Returns:
        measure (list): 逐元素对应逐行评估指标输出
    """
    measure = []
    for n in N:
        measure.append('Top ' + str(n) + '\n')
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        indicators = []
        if len(origin) != len(predicted):
            print('The Lengths of test set and predicted set do not match!')
            exit(-1)
        hits = Metric.hits(origin, predicted)
        hr = Metric.hit_ratio(origin, hits)
        indicators.append('Hit Ratio:' + str(hr) + '\n')
        prec = Metric.precision(hits, n)
        indicators.append('Precision:' + str(prec) + '\n')
        recall = Metric.recall(hits, origin)
        indicators.append('Recall:' + str(recall) + '\n')
        NDCG = Metric.NDCG(origin, predicted, n)
        indicators.append('NDCG:' + str(NDCG) + '\n')
        measure += indicators
    return measure

def rating_evaluation(res):
    measure = []
    mae = Metric.MAE(res)
    measure.append('MAE:' + str(mae) + '\n')
    rmse = Metric.RMSE(res)
    measure.append('RMSE:' + str(rmse) + '\n')
    return measure