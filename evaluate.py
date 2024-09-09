from model.EnvAgentCreate import create_generate, create_improve
from model.GenerateImproveModel import GIModel
import numpy as np
import cv2

file_path = '../dataset/1005test/'
model_path = "../fjsp_drl_main/model/save_10_5.pt"
generate_env, generate_agent = create_generate(file_path, model_path)
improve_env, improve_agent = create_improve()
for env_index in range(len(improve_env.graph_list)):
    agent = GIModel(improve_env, generate_env[env_index], improve_agent, generate_agent)
    schedule = np.array([])
    images = []
    ope_num = agent.improve_env.graph_list[env_index].ope_num
    for i in range(ope_num):
        solution = agent.generate(schedule)
        if i == ope_num - 1:
            schedule = agent.improve(solution, improve_times=ope_num, env_index=env_index)
        else:
            schedule = agent.improve(solution, improve_times=0, env_index=env_index)
        # graph = copy.deepcopy(improve_env.graph_list[0])  # 此处将要画的solution复制过来
        # graph.apply_solution(solution)
        # graph.cal_sw_tw()
        # cmax = graph.makespan()
        # gantchart = GanttChart(graph)
        image = agent.improve_env.render()
        images.append(image)

    width, height = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 指定编码器
    video = cv2.VideoWriter('render_result/output_' + str(env_index) + '.mp4', fourcc, 2, (width, height))
    for img in images:
        # 将 PIL 图像转换为 numpy 数组
        frame = np.array(img)
        # 将 RGB 转换为 BGR（OpenCV 使用 BGR）
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 写入视频
        video.write(frame)

    # 释放 VideoWriter 对象
    video.release()

    print(agent.improve_env.graph.makespan())
