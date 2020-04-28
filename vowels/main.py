import vowels_task1 as vt1
import vowels_task2 as vt2
import latexconfusiontable as lct


def main():
    print("Task 1")
    conf1 = vt1.train_test_single_gaussian(0,70,False)
    #lct.print_confusion(conf1)
    print("------------------------------------------")
    print("Task 2")
    conf2 = vt2.train_test_GMM(0,70,2)
    #lct.print_confusion(conf2)


main()