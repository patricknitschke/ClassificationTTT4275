import vowels_task1 as vt1
import vowels_task2 as vt2


def main():
    print("Task 1")
    vt1.train_test_single_gaussian(0,70,False)
    print("------------------------------------------")
    print("Task 2")
    vt2.train_test_GMM(0,70,2)


main()