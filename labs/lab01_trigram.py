from collections import Counter, defaultdict

cau = ["hôm nay trời đẹp", "hôm nay trời mưa", "hôm qua trời mưa"]

trigram = defaultdict(Counter)   # trigram[(a, b)][c] = số lần c đứng sau cặp (a, b)
for s in cau:
    tu = ["<s>", "<s>"] + s.split()
    for a, b, c in zip(tu, tu[1:], tu[2:]):
        trigram[(a, b)][c] += 1

def top3(a, b):
    dem = trigram[(a, b)]
    tong = sum(dem.values())
    return [(w, round(n / tong, 2)) for w, n in dem.most_common(3)]

print(top3("hôm", "nay"))    # [('trời', 1.0)]
print(top3("nay", "trời"))   # [('đẹp', 0.5), ('mưa', 0.5)]
print(top3("qua", "trời"))   # [('mưa', 1.0)]

# Tính xác suất cả câu "hôm nay trời mưa" bằng cách nhân các xác suất trigram.
def p_trigram(a, b, c):
    dem = trigram[(a, b)]
    return dem[c] / sum(dem.values())

def p_cau(s):
    tu = ["<s>"] + s.split()
    p = 1.0
    for a, b, c in zip(tu, tu[1:], tu[2:]):
        p *= p_trigram(a, b, c)
    return p

print(p_cau("hôm nay trời mưa"))   # 0.3333...