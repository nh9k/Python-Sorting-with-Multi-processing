"""
자료구조 및 알고리즘 
Homework#1
21611591 김난희
"""

import random                    # random 메소드 사용을 위함
import time                      # time을 구함
import numpy                     # mean값과 std 값을 구함
from multiprocessing import Pool # cpu multiprocessing
from math import log             # Radix sort에서 사용하기 위함


def swap(x, i, j):               # swap function
    x[i], x[j] = x[j], x[i]      # 원소 교환


# * * * * *  * * * * * Bubble Sort (버블정렬) * * * * *  * * * * * # O(N^2)
def bubbleSort(x):                       # original bubble sort: short bubble이 아닌 원본 버전
    for size in reversed(range(len(x))): # 크기를 하나씩 줄여가며 반복
        for i in range(size):            # 리스트의 크기만큼 반복
            if x[i] > x[i + 1]:          # 현재의 index 값이 다음 index의 값보다 크면 참 (비교: N-1,N-2,...1)
                swap(x, i, i + 1)        # 위치 교환


# * * * * *  * * * * * Selection Sort (선택정렬) * * * * *  * * * * * # O(N^2)
def selectionSort(x):
    for size in reversed(range(len(x))): # max 수합하면 하나 제외하고 찾는 for문
        max_i = 0                        # max 초기값 0
        for i in range(1, 1+size):       # max 찾는 for문 # max 수합 후 범위에서 제외하고 다시 범위를 줌
            if x[i] > x[max_i]:          # 현재 index 값이 max로 되어있는 값보다 크다면 (비교: N-1번)
                max_i = i                # max에는 현재 index를 넣어줌
        swap(x, max_i, size)             # max를 수합하여 이미 정렬된 것중 제일 앞에 수와 수합 (수합: 1번)


# * * * * *  * * * * * Insertion Sort (삽입정렬) * * * * *  * * * * * # O(N^2) 
def insertionSort(x):                                                 # 통계적으로 bubble이나 selection보다 비교연산이 적음
    for size in range(1, len(x)):        # 리스트 크기만큼 반복
        val = x[size]                    # 처음 index 부터 시작함 # 현재 읽고 있는 값
        i = size                         # 현재 읽고 있는 index
        while i > 0 and x[i-1] > val:    # 첫 번째 index 다음 부터 시작 # 현재 값이 이전 값보다 크면
            x[i] = x[i-1]                # 현재 값에 이전 값을 넣고
            i -= 1                       # index를 하나 감소 시켜서 # 또다시 이전 값과 비교하여 while문 반복
        x[i] = val                       # 크기가 알맞은 위치에 삽입


# * * * * *  * * * * *  Shell Sort (쉘 정렬) * * * * *  * * * * * # average O(N^1.5)
def InsertionSort(x, start, gap):                # 삽입정렬 구현
    for target in range(start+gap, len(x), gap): # (시작인덱스+차이, 리스트 크기만큼 반복, 차이까지)
        val = x[target]                          # 리스트의 값
        i = target                               # 인덱스 저장
        while i > start:                         # 증감 값 보다 인덱스가 크다면 반복
            if x[i-gap] > val:                   # 리스트의 비교 인덱스 값 보다 크다면
                x[i] = x[i-gap]                  # 해당 인덱스 값 할당
            else:                                # 리스트의 비교 인덱스 값 보다 작다면
                break                            # 반복 중지
            i -= gap                             # 중간 값만큼 빼주기
        x[i] = val                               # 해당 값 삽입

def shellSort(x):                        # 본 shell sort
    gap = len(x) // 2                    # 리스트를 2로 나눈 몫 (중간 값) 취함
    while gap > 0:                       
        for start in range(gap):         # 중간 값의 크기만큼 반복
            InsertionSort(x, start, gap) # 삽입정렬 메소드 호출 (리스트, 증감 값, 중간 값)
        gap = gap // 2                   # 리스트를 2로 나눈 몫 (중간 값) 취함 (반으로 줄여나간다.)


# * * * * *  * * * * * Merge Sort (병합 정렬) * * * * *  * * * * * # O(nlog(n))
def mergeSort(x): 
    if len(x) > 1:                  # 배열의 길이가 1보다 클 경우 재귀함수 호출 반복
        mid = len(x)// 2            # 2로 나눈 몫 (중간 값) 취함
        lx, rx = x[:mid], x[mid:]   # 중간 기준으로 왼쪽(lx), 오른쪽(rx) split
        mergeSort(lx)               # left sublist의 값을 기준으로 병합 정렬 재귀 호출         
        mergeSort(rx)               # right sublist의 값을 기준으로 병합정렬 재귀 호출

        li, ri, i = 0, 0, 0                  # left sublist, right sublist, merged list
        while li < len(lx) and ri < len(rx): # 두 sublist 길이만큼 돌린다. # sublist의 정렬이 끝날 때 까지
            if lx[li] < rx[ri]:              # left sublist의 현재 값이 right sublist의 현재 값보다 작다면
                x[i] = lx[li]                # left sublist의 현재 값을 merged list에 넣음
                li += 1                      # left sublist의 index 증가(다음 값 검사를 위함)
            else:                            # left sublist의 현재 값이 right sublist의 현재 값보다 크다면
                x[i] = rx[ri]                # right sublist의 현재 값을 merged list에 넣음
                ri += 1                      # right sublist의 index 증가(다음 값 검사를 위함)
            i += 1                           # merged list에 값을 넣은 후 다음 값을 넣기 위해 index 증가
        x[i:] = lx[li:] if li != len(lx) else rx[ri:] # 두 sublist 중 모든 원소가 merged list에 들어가서 더 이상 비교할 것이 없다면 
                                                      # 남은 원소가 있는 sublist를 다시 x에 넣고 merge sort함


# * * * * *  * * * * * Quick Sort (퀵 정렬) * * * * *  * * * * * # Average: O(nlog(n), Worst(이미 정렬된 list): O(n^2)
def pivotFirst(x, lmark, rmark):                        # left, right mark를 둔 method
    pivot_val = x[lmark]                                # left mark(첫 index부터 시작)의 값을 pivot value으로
    pivot_idx = lmark                                   # left mark(첫 index부터 시작)의 index를 pivot index로
    while lmark <= rmark:                               # left와 right mark가 cross 되기 전까지
        while lmark <= rmark and x[lmark] <= pivot_val: # pivot value보다 작으면 넘어감 # pivot value보다 큰 수 찾음
            lmark += 1                                  # index를 증가시켜 넘어감(오른쪽으로)
        while lmark <= rmark and x[rmark] >= pivot_val: # pivot value보다 크면 넘어감 # pivot value보다 작은 수 찾음
            rmark -= 1                                  # index를 감소시켜 넘어감(왼쪽으로)
        if lmark <= rmark:                              # left가 pivot보다 큰 수 찾고, right가 pivot보다 작은 수 찾고, no cross 이면
            swap(x, lmark, rmark)                       # left와 right mark의 값 수합
            lmark += 1                                  # index 다음으로 넘어가며 검사
            rmark -= 1
    swap(x, pivot_idx, rmark)                           # left와 right mark가 cross한 상황에서 pivot과 right mark 수합
    return rmark                                        # 수합한 right mark index 위치 반환

def quickSort(x, pivotMethod=pivotFirst):               # left, right mark를 둔 method를 사용하여 퀵 정렬
    def _qsort(x, first, last):
        if first < last:
            splitpoint = pivotMethod(x, first, last)    # 위의 pivotFirst 함수를 통해 split point(수합한 right mark 위치)를 정함
            _qsort(x, first, splitpoint-1)              # 재귀 호출 (리스트, 시작 인덱스, 수합한 수index-1) 왼쪽 절반
            _qsort(x, splitpoint+1, last)               # 재귀 호출 (리스트, 수합한 수index+1, 종료인덱스) 오른쪽 절반
    _qsort(x, 0, len(x)-1)                              # _qsort 함수 호출


# * * * * *  * * * * *  Radix Sort (기수 정렬) * * * * *  * * * * * # O(nlog(n)), ideal: O(n)
def get_digit(number, d, base):                   ## 현재 자릿수(d)와 진법(base)에 맞는 숫자 변환   
  return (number // base ** d) % base             # //는 몫 연산자(그 다음 연산) **는 지수 연산자(우선 연산)
                                                  # ex) number = 102, d = 0, base = 10 -> 첫 번째 자리수에  해당하는 2 찾음
def counting_sort_with_digit(A, d, base):         ## 자릿수 기준으로 counting sort  # A: input array  # d : 현재 자릿수 # ex) 102, d = 0 : 2                                                                                            
    k = base - 1                                  # k: maximum value of A # ex) 10진수의 최대값 = 9
    B = [-1] * len(A)                             # B: output array    # init with -1
    C = [0] * (k + 1)                             # C: count array  # init with zeros
      
    for a in A:                                   ## 현재 자릿수를 기준으로 빈도수 세기
        C[get_digit(a, d, base)] += 1             # 숫자를 하나씩 카운팅
    for i in range(k):                            ## C 업데이트
        C[i + 1] += C[i]                          # 각 요소값에 직전 요소값을 더함
    for j in reversed(range(len(A))):             ## 현재 자릿수를 기준으로 정렬 
        B[C[get_digit(A[j], d, base)] - 1] = A[j] # A 요소값의 역순으로 B를 채워 넣음
        C[get_digit(A[j], d, base)] -= 1          # 자리를 채운 후 해당하는 값에서 1 소모
    return B

def radixSort(list, base=10):     
    digit = int(log(max(list), base) + 1) # 입력된 리스트 가운데 최대값의 자릿수(digit) 확인   
    for d in range(digit):                # 자릿수 별로 counting sort
        list = counting_sort_with_digit(list, d, base)
    return list


# * * * * *  * * * * *  Heap Sort (힙 정렬) * * * * *  * * * * * # O(nlog(n))
def heapify(x, index, heap_size):                               # heap 구조를 유지하는 함수 # heap 성질 만족하도록
    largest = index                                             # 배열의 중간부터 시작
    left_index = 2 * index + 1                                  # 왼쪽 자식노드의 index를 left_index 
    right_index = 2 * index + 2                                 # 오른쪽 자식노드의 index를 right_index
    if left_index < heap_size and x[left_index] > x[largest]:   # 왼쪽 자식 노드 값 > 부모 노드 값
        largest = left_index                                    # 왼쪽 자식 노드 index를 largest에 저장
    if right_index < heap_size and x[right_index] > x[largest]: # 오른쪽 자식 노드 값 > 부모 노드 값
        largest = right_index                                   # 오른쪽 자식 노드 index를 largest에 저장
    if largest != index:                                        # 본래 부모 노드 index와 달라졌다면
        swap(x, largest, index)                                 # 위치를 바꿈
        heapify(x, largest, heap_size)                          # 바꾼 자식 노드에서 다시 힙 성질 유지하는지 검사

def heapSort(x): 
    n = len(x)

    ### BUILD-MAX-HEAP: 주어진 원소들로 최대 힙 구성  
    # 최초 힙 구성시 배열의 중간부터 시작하면 
    # 이진트리 성질에 의해 모든 요소값을 서로 한번씩 비교할 수 있게 됨 : O(n) 
    for i in range(n // 2 - 1, -1, -1): # 위에서 아래로 heapify 수행    # index : (n을 2로 나눈 몫-1)~0
        heapify(x, i, n)

    ### Recurrent: max heap의 root(최댓값)과 마지막 요소 교환 + 새 root 노드에 대한 max heap 구성
    # 한번 힙이 구성되면 개별 노드는 최악의 경우에도 트리의 높이(log(n)) 만큼의 자리 이동을 하게 됨
    # 이런 노드들이 n개 있으므로 : O(nlog(n))
    for i in range(n - 1, 0, -1):
        swap(x, 0, i)           
        heapify(x, 0, i)
    return x

def randomNum(k):                                                # 0부터 10의 k승까지 10의 k승 숫자를 random으로 뽑음
    num_list = random.sample(range(0, 10**k+1), 10**k+1)         # 10**k는 pow(10,k)로도 나타낼 수 있음, 10^k 의미함
    #print(num_list)
    return num_list                                              # 랜덤한 수가 담긴 list 반환

def testFunc(args):             # 인자: 함수, data 수의 지수, base(0은 radix sort 외, 0 제외 숫자는 radix sort의 base)
    Sorting_func_call = args[0] # 함수를 인자로 받아옴
    k = args[1]                 # N=10^k, data 개수
    base = args[2]              # radix 정렬의 base = 8, 16, 32 ...

    time_list=[]                # 계산한 elapsed time의 리스트, 평균과 표준 편차를 계산하기 위함

    if base==0: #insertion, bubble, selection, shell, merge, quick, heap sort실행
        for i in range(10):
            start_time = time.time()            # 시작 시간
            num_list=randomNum(k)               # random한 숫자 N=10^k, [0..N]
            Sorting_func_call(num_list)         # Sorting function
            #print(num_list)                    # sorting 되었는지 확인
            end_time=(time.time() - start_time) # 실행(elapsed) 시간 = 현재 시간 - 시작 시간
            #print(end_time)
            time_list.append(end_time)          # 시간을 리스트로 append

        print("input data: 10^%s -> %18s -> elapsed time mean: %s seconds" 
              % (k, Sorting_func_call.__name__, numpy.mean(time_list)))     # 10개 test 값의 평균
        print("input data: 10^%s -> %18s -> elapsed time  std: %s seconds" 
              % (k, Sorting_func_call.__name__,  numpy.std(time_list)))     # 10개 test 값의 표준편차

    else: #radixSort(unsorted list, base = 8, 16, 32 ...)
        for i in range(10):
            start_time = time.time()                  #시작 시간
            num_list=randomNum(k)                     # random한 숫자 N=10^k, [0..N]
            num_list=Sorting_func_call(num_list,base) # Sorting function
            #print(num_list)                          # sorting 되었는지 확인
            end_time=(time.time() - start_time)       # 실행(elapsed) 시간 = 현재 시간 - 시작 시간
            time_list.append(end_time)                # 시간을 리스트로 append

        print("input data: 10^%s -> %8s(base %2s) -> elapsed time mean: %s seconds" 
              % (k, Sorting_func_call.__name__, base, numpy.mean(time_list)))   # 10개 test 값의 평균
        print("input data: 10^%s -> %8s(base %2s) -> elapsed time  std: %s seconds" 
              % (k, Sorting_func_call.__name__, base,  numpy.std(time_list)))   # 10개 test 값의 표준편차
    
def main():                  # multiprocessing 위주의 구현 # test function은 따로 분리(testFunc(args))
    pool = Pool(processes=4) # cpu 4 core 사용 
    for k in range(2,6):     # input data: 10^2, 10^3, 10^4, 10^5 
        pool.map(testFunc,[[bubbleSort,k,0],[insertionSort,k,0],[selectionSort,k,0],         # cpu에게 일을 던져줌
                           [shellSort,k,0],[mergeSort,k,0],[quickSort,k,0],
                           [heapSort,k,0],[radixSort,k,8],[radixSort,k,16],[radixSort,k,32]]) 
        # pool.map(testFunc,[[bubbleSort,k,0]]) # test 1 sorting method
    pool.close()    # 병렬 처리가 끝났을 때, 프로세스 종료
    pool.join()     # 작업자 프로세스가 종료 될 때까지 기다림

if __name__ == '__main__':
    main()                  # main()문 실행