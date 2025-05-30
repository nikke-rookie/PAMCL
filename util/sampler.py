from random import choice
from datetime import datetime
from data.ui_graph import Interaction

def next_batch_pairwise(data: Interaction, batch_size, n_negs):
    """
    生成用于训练的批量样本对

    Args:
        data (Interaction): 模型数据
        batch_size: 批量大小，决定每次迭代返回的样本数量
        n_negs: 每个用户的负样本数量

    Returns:
        yield: 每次yield出包含用户id、正样本id和负样本id列表
    """

    if n_negs <= 0:
        raise ValueError("n_negs must be greater than 0")
    training_data = data.training_data

    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size

        batch_users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        batch_items = [training_data[idx][1] for idx in range(ptr, batch_end)]

        ptr = batch_end
        u_ids: list[int] = []
        i_ids: list[int] = []
        j_ids: list[list[int]] = []

        item_list = list(data.item.keys())

        for i, user in enumerate(batch_users):
            i_ids.append(data.get_item_id(batch_items[i]))
            u_ids.append(data.get_user_id(user))

            # 生成指定数量的负样本索引，并添加到j_ids
            neg_items: list[str] = []
            for _ in range(2*n_negs):
                neg_item = choice(item_list)
                while neg_item in data.training_set_u[user]:
                    neg_item = choice(item_list)
                neg_items.append(neg_item)

            j_ids.append([data.get_item_id(item) for item in neg_items])

        yield u_ids, i_ids, j_ids
