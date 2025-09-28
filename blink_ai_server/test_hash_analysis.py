#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def hamming_distance(hash1, hash2):
    """计算两个hash值之间的汉明距离"""
    if len(hash1) != len(hash2):
        return float('inf')
    
    distance = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            distance += 1
    
    return distance

def analyze_hashes():
    """分析hash值的相似性"""
    hashes = {
        'person2_2': '01011111110111001110001111100011',
        'person2_3': '00011001011111001101101111000001',
        'person2_1_face5': '01111001110011001011100101000001'
    }
    
    print("Hash值分析")
    print("=" * 50)
    
    # 显示所有hash值
    for name, hash_val in hashes.items():
        print(f"{name}: {hash_val}")
    
    print("\n汉明距离分析:")
    print("-" * 30)
    
    # 计算两两之间的汉明距离
    names = list(hashes.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            hash1, hash2 = hashes[name1], hashes[name2]
            distance = hamming_distance(hash1, hash2)
            print(f"{name1} vs {name2}: {distance}")

if __name__ == "__main__":
    analyze_hashes()
