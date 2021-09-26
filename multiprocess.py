from multiprocessing import Pool, cpu_count
import time
from timeit import default_timer as timer

def get_perf (template_path, directives_path, top_function, part, parameters, verbose=False, timelimit=500):
    print(template_path)
    print(directives_path)
    print(top_function)
    print(part)
    print(parameters)
    print(timelimit)

def main():
    start = timer()

    print(f'starting computations on {cpu_count()} cores')

    parameters = [1,2,3,4,5]
    args = []
    for p in parameters:
        arg = [
            'tpath',
            'dpath',
            'tfile',
            'part',
            p
        ]
        args.append(arg)

    with Pool() as pool:
        res = pool.starmap(get_perf, args)
        print(res)

    end = timer()
    print(f'elapsed time: {end - start}')


if __name__ == '__main__':
    main()