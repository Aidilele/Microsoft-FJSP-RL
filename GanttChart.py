import matplotlib.pyplot as plt
from UsefulFunction import get_color, encoding_color_list2str
from Graph import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import io
import imageio


class GanttChart:

    def __init__(self, graph):
        self.graph = graph
        self.font_style_task = {
            "family": "Microsoft YaHei",
            "style": "oblique",
            "weight": "bold",
            "color": "black",
            "size": 2
        }

        color = get_color(self.graph.job_num)
        self.color = []
        for each_color in color:
            color_str = encoding_color_list2str(each_color)
            self.color.append(color_str)
        self.color = tuple(self.color)

    def get_gantt_struct_data(self):
        struct_data = {}
        solution = self.graph.solution
        self.graph.cal_sw_tw()
        for machine in solution:
            for ope in machine:
                operation = self.graph.trans2object(ope)
                operation_keys = (
                    operation.location[0],
                    operation.location[1],
                    operation.process_machine)
                operation_values = (
                    operation.s_w,
                    operation.s_w + operation.p_t,
                    operation.p_t,
                )
                struct_data[operation_keys] = operation_values
        return struct_data

    def plot(self):

        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 显示中文标签
        struct_data = self.get_gantt_struct_data()

        plt_list = []
        plt_name_list = []
        name = 0
        for each_color in self.color:
            each_plt = plt.barh(0, 0, 0, 0, color=each_color)
            plt_list.append(each_plt)
            plt_name_list.append('job ' + str(name))
            name += 1
        plt.legend(plt_list, plt_name_list, loc='best', bbox_to_anchor=(1.01, 1))
        for k, v in struct_data.items():
            plt.barh(y=k[2], height=0.8, width=v[2], left=v[0], edgecolor="black", color=self.color[k[0]])
            # plt.text(v[0], 2 * k[2], str(k[0]) + "-" + str(k[1]), fontdict=self.font_style_task)

        ylabels = ['', ]  # 生成y轴标签
        for machine_index in range(self.graph.machine_num):
            ylabels.append('Machine' + str(machine_index))

        plt.yticks(range(-1, self.graph.machine_num), ylabels, rotation=0)
        plt.title("Gantt Chart")
        plt.xlabel("Process Time /min")

        buf = io.BytesIO()
        canvas = FigureCanvas(plt.gcf())
        canvas.print_png(buf)
        buf.seek(0)
        image = Image.open(buf)
        # array = np.array(image)
        plt.close()
        # plt.show()
        return image


def main_gantt_chart(graph):
    gantchart = GanttChart(graph)
    gantchart.plot()


if __name__ == '__main__':
    file_path = './dataset/1005/3128_10j_5m.fjs'  # 此处输入对应instance的文件路径
    graph = FJSPGraph(file_path)
    solution = 0  # 此处将要画的solution复制过来
    graph.apply_solution(solution)
    graph.cal_sw_tw()
    cmax = graph.makespan()
    gantchart = GanttChart(graph)
    gantchart.plot()
