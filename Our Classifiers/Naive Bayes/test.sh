echo KYOTO
python3 naive_bayes.py BASE kyoto_base KYOTO --posterior
python3 naive_bayes.py PREV_CLASS kyoto_class KYOTO
python3 naive_bayes.py TIME_DEPENDENCY kyoto_time KYOTO
python3 naive_bayes.py MI kyoto_mi KYOTO
python3 naive_bayes.py MI_EXT kyoto_mi_ext KYOTO

echo ARUBA
python3 naive_bayes.py BASE aruba_base ARUBA
python3 naive_bayes.py PREV_CLASS aruba_class ARUBA
python3 naive_bayes.py TIME_DEPENDENCY aruba_time ARUBA
python3 naive_bayes.py MI aruba_mi ARUBA
python3 naive_bayes.py MI_EXT aruba_mi_ext ARUBA

echo ARUBAEXT
python3 naive_bayes.py BASE arubaext_base ARUBAEXT
python3 naive_bayes.py PREV_CLASS arubaext_class ARUBAEXT
python3 naive_bayes.py TIME_DEPENDENCY arubaext_time ARUBAEXT
python3 naive_bayes.py MI arubaext_mi ARUBAEXT
python3 naive_bayes.py MI_EXT arubaext_mi_ext ARUBAEXT