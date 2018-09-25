def intersect(nums1, nums2):
    record, result = {}, []
    for num in nums1:
        record[num] = record.get(num, 0) + 1

    for num in nums2:
        if num in record and record[num]:
            result.append(num)
            record[num] -= 1
    return result