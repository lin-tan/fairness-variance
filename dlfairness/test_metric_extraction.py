import traceback
import training_code_metric_processing


def main():
    fl_file = open('source_files_list.txt', 'r')
    files_list = fl_file.readlines()
    fl_file.close()

    metric_out = open('extracted_metrics.csv', 'w')
    metric_out.write('Files, Metrics\n')

    for file in files_list:
        file = file.rstrip()
        # if file == 'INQ-pytorch/examples/imagenet_quantized.py':
        try:
            filename = 'source_training_files/' + file
            main_loops = training_code_metric_processing.extract_metric(filename)
            metric_out.write('\n%s,\n' % filename)

            for loop in main_loops:
                (node, loop_var, loop_line, funcs, NI_metrics) = loop
                metric_out.write('Loop,%s,%d\n' % (loop_var, loop_line))

                for func in funcs:
                    (func_node, funcn, func_line, metrics) = func

                    if len(metrics) == 0:
                        continue

                    metric_out.write(',Func,%s,%d\n' % (funcn, func_line))

                    for metric in metrics:
                        (metric_node, metric, metric_line) = metric
                        metric_out.write(',,Metric,%s,%d\n' % (metric, metric_line))

                for metric in NI_metrics:
                    metric_out.write(',NI_Metric,%s\n' % metric)

        except Exception:
            print(filename)
            print(traceback.format_exc())

    metric_out.close()


if __name__ == "__main__":
    main()
