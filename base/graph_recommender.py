from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
import os
from os.path import abspath
from util.evaluation import ranking_evaluation
from dotenv import load_dotenv
from qywx_bot.bot import Bot

load_dotenv()
key = os.getenv('WEBHOOK_KEY')
if key is not None:
    bot: Bot = Bot(key)


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super().__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set, **kwargs)
        self.early_stop = 0
        self.bestPerformance = []
        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super().print_model_info()
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def predict(self, u):
        raise NotImplementedError

    def test(self, pre_trained=False, file: str = ""):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            print(f'\rProgress: [{"+" * ratenum}{" " * (50 - ratenum)}]{ratenum * 2}%', end='', flush=True)

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user, pre_trained=pre_trained, file=file)
            rated_list, _ = self.data.user_rated(user)
            # 根据用户历史评分，排除已评分项目(赋极小值)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list, pre_trained=False):
        #* 仅在所有训练epoch结束后执行一次
        """
        输出推荐指标及结果

        Args:
            rec_list (dict): 推荐列表 `{user: [(item1, score1), (item2, score2), ...]}`
        """
        self.recOutput.append('userId: recommendations in (itemId, ranking score) pairs, * means the item is hit.\n')
        # 结果输出样例
        # 0: (771,3.333857297897339) (649,3.1552059650421143) (...)
        for user in self.data.test_set:
            line = user + ':' + ''.join(
                f" ({item[0]},{item[1]}){'*' if item[0] in self.data.test_set[user] else ''}"
                for item in rec_list[user]
            )
            line += '\n'
            self.recOutput.append(line)
        
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output

        top_items_file = f"{self.config['model']['name']}@{current_time}-top-{self.max_N}items.txt"
        performance_file = f"{self.config['model']['name']}@{current_time}-performance.txt"

        self.result = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        # self.model_log.add('###Evaluation Results###')
        # result_format_str = '\n'
        # for r in self.result:
        #     result_format_str += r
        # self.model_log.add(result_format_str)
        self.model_log.add(f"The result of {self.model_name}: {''.join(self.result)}")

        end_time = time()
        train_time = end_time - self.start_time
        self.model_log.add(f"Run time: {train_time:.2f}s")

        if not pre_trained:
            FileIO.write_file(out_dir, top_items_file, self.recOutput)
            self.model_log.add(f"Top {self.max_N} items have been output to \"{abspath(out_dir)}/{top_items_file}\"")
            FileIO.write_file(out_dir, performance_file, self.result)
            self.model_log.add(f"Performance result has been output to \"{abspath(out_dir)}/{performance_file}\"")

        # bot.send_text(f'[whr] The result of {self.model_name}:\n{"".join(self.result)}\nRun time: {train_time:.2f}s')

    def fast_evaluation(self, epoch):
        """
        输出单轮评估指标并记录最佳性能

        Returns:
            measure (list): 逐元素对应逐行评估指标输出
        """
        self.model_log.add('Evaluating the model...')
        rec_list = self.test()
        measure = ranking_evaluation(self.data.test_set, rec_list, [self.max_N])
        # 从1开始, 0是标题
        performance = {k: float(v) for m in measure[1:] for k, v in [m.strip().split(':')]}

        # 如果存在之前的最佳性能记录
        if len(self.bestPerformance) > 0:
            count = 0
            performance = {}
            for m in measure[1:]:
                k, v = m.strip().split(':')
                performance[k] = float(v)
            # 比较当前性能和最佳性能，更新count值
            for k in self.bestPerformance[1]:
                if self.bestPerformance[1][k] > performance[k]:
                    count += 1
                else:
                    count -= 1
            # 通过count值粗略判断是否整体更优(即多数指标更优)
            if count < 0:
                self.early_stop = 0  # 重新置为0
                self.bestPerformance = [epoch + 1, performance]
                self.save()
            else:
                self.early_stop += 1  # 性能未更新+1
        else:
            # 不存在历史最佳性能记录，则直接保存
            self.bestPerformance = [epoch + 1, performance]
            self.save()

        measure_str = ', '.join([f'{k}: {v}' for k, v in performance.items()])
        self.model_log.add(f'*Current Performance* 💡 Epoch: {epoch + 1}, {measure_str}')
        bp = ', '.join([f'{k}: {v}' for k, v in self.bestPerformance[1].items()])
        self.model_log.add(f'*Best Performance* 🔥 Epoch: {self.bestPerformance[0]}, {bp}')
        return measure
