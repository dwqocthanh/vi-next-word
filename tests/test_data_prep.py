import unicodedata
 
from src.data_prep import clean_line, split_files
 
 
def test_tach_chu_dinh_sau_dau_cham():
    assert clean_line("ảm đạm.gió mưa") == "ảm đạm. gió mưa"
 
 
def test_chuan_hoa_nfc():
    roi = unicodedata.normalize("NFD", "ế")      # chữ ế ở dạng "rời"
    assert clean_line(roi) == "ế"
    assert unicodedata.is_normalized("NFC", clean_line(roi))
 
 
def test_khong_dinh_so_thap_phan():
    assert clean_line("giá 3.5 đồng") == "giá 3.5 đồng"
 
 
def test_chia_tap_khong_giao_nhau():
    files = [f"f{i}.txt" for i in range(100)]
    p = split_files(files)
    tat_ca = p["train"] + p["val"] + p["test"]
    assert len(tat_ca) == len(set(tat_ca)) == 100
    assert len(p["test"]) == 5 and len(p["val"]) == 5
