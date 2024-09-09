def single_process(file_path='./dataset/1005/3128_10j_5m.fjs', dict_mode=True):
    file = open(file_path)
    doc = []
    for line in file.readlines():
        doc.append(line[:-1])
    job_num, machine_num, _ = readstrline(doc[0])
    struct_data = []
    ope_total = 0
    for i in range(job_num):
        raw_job_data = readstrline(doc[i + 1])
        index = 0
        ope_num = raw_job_data[index]
        index += 1
        struct_job_data = []
        for j in range(ope_num):
            ope_mach_pair_num = raw_job_data[index]
            index += 1
            struct_pair_data = [-1] * (machine_num + 1)
            ope_total += 1
            for k in range(ope_mach_pair_num):
                struct_pair_data[raw_job_data[index]] = raw_job_data[index + 1]
                index += 2
            struct_job_data.append(struct_pair_data[1:])
        struct_data.append(struct_job_data)
    if dict_mode:
        for each_job in range(len(struct_data)):
            for each_ope in range(len(struct_data[each_job])):
                pair_dict = {}
                for each_pair in range(len(struct_data[each_job][each_ope])):
                    if struct_data[each_job][each_ope][each_pair] != -1:
                        pair_dict[each_pair] = struct_data[each_job][each_ope][each_pair]
                struct_data[each_job][each_ope] = pair_dict

    return struct_data, machine_num, job_num, ope_total


def readstrline(s):
    result = []
    num = ''
    for each in s:
        if each != ' ' and each != '\t':
            num += each
        else:
            if num != '':
                result.append(int(num))
                num = ''
    if num != '': result.append(int(float(num)))
    return result


if __name__ == '__main__':
    single_process(dict_mode=False)
