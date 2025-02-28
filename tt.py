def getMinPerformers(startTime, endTime):
    # 将开始时间和结束时间打包在一起，然后根据结束时间进行排序
    performances = sorted(zip(startTime, endTime), key=lambda x: (x[1], x[0]))

    # 最少需要的表演者数量
    required_performers = 0
    # 当前表演结束时间，初始化为非常小的值
    current_end_time = -1

    # 遍历所有表演
    for start, end in performances:
        # 如果当前表演的开始时间在上一个表演结束之后，不需要额外的表演者
        if start > current_end_time:
            # 新增一个表演者，更新当前表演的结束时间
            required_performers += 1
            current_end_time = end

    return required_performers

# 示例调用
startTime = [1, 4, 5]
endTime = [5, 5, 6]
print(getMinPerformers(startTime, endTime))  # 应输出表演者的最小数量