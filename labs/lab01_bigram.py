from collections import Counter, defaultdict
 
cau = ["hôm nay trời đẹp", "hôm nay trời mưa", "hôm qua trời mưa"]
 
bigram = defaultdict(Counter)           # bigram[a][b] = số lần b đứng sau a
for s in cau:
    tu = ["<s>"] + s.split()
    for a, b in zip(tu, tu[1:]):
        bigram[a][b] += 1
 
 
def top3(tu_truoc):
    dem = bigram[tu_truoc]
    tong = sum(dem.values())
    return [(w, round(n / tong, 2)) for w, n in dem.most_common(3)]
 
 
print(top3("trời"))      # [('mưa', 0.67), ('đẹp', 0.33)]
print(top3("hôm"))       # [('nay', 0.67), ('qua', 0.33)]

# Tính xác suất cả câu "hôm nay trời mưa" bằng cách nhân các xác suất bigram.
def p_bigram(a, b):
    dem = bigram[a]
    return dem[b] / sum(dem.values())

def p_cau(s):
    tu = ["<s>"] + s.split()
    p = 1.0
    for a, b in zip(tu, tu[1:]):
        p *= p_bigram(a, b)
    return p

print(p_cau("hôm nay trời mưa"))   # 0.4444...