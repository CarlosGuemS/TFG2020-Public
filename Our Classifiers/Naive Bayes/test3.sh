echo KYOTO1
python3 naive_bayes.py BASE kyoto1_base KYOTO1 --posterior
python3 naive_bayes.py PWC kyoto1_class KYOTO1
python3 naive_bayes.py TD kyoto1_time KYOTO1
python3 naive_bayes.py MI kyoto1_mi KYOTO1
python3 naive_bayes.py EMI kyoto1_EMI KYOTO1
python3 naive_bayes.py PWC+TD kyoto1_PWC_TD KYOTO1
python3 naive_bayes.py PWC+EMI kyoto1_PWC_EMI KYOTO1
python3 naive_bayes.py TD+EMI kyoto1_TD_EMI KYOTO1
python3 naive_bayes.py PWC+TD+EMI kyoto1_PWC_TD_EMI KYOTO1

echo KYOTO2
python3 naive_bayes.py BASE kyoto2_base KYOTO2
python3 naive_bayes.py PWC kyoto2_class KYOTO2
python3 naive_bayes.py TD kyoto2_time KYOTO2
python3 naive_bayes.py MI kyoto2_mi KYOTO2
python3 naive_bayes.py EMI kyoto2_EMI KYOTO2
python3 naive_bayes.py PWC+TD kyoto2_PWC_TD KYOTO2
python3 naive_bayes.py PWC+EMI kyoto2_PWC_EMI KYOTO2
python3 naive_bayes.py TD+EMI kyoto2_TD_EMI KYOTO2
python3 naive_bayes.py PWC+TD+EMI kyoto2_PWC_TD_EMI KYOTO2

echo KYOTO3
python3 naive_bayes.py BASE kyoto3_base KYOTO3
python3 naive_bayes.py PWC kyoto3_class KYOTO3
python3 naive_bayes.py TD kyoto3_time KYOTO3
python3 naive_bayes.py MI kyoto3_mi KYOTO3
python3 naive_bayes.py EMI kyoto3_EMI KYOTO3
python3 naive_bayes.py PWC+TD kyoto3_PWC_TD KYOTO3
python3 naive_bayes.py PWC+EMI kyoto3_PWC_EMI KYOTO3
python3 naive_bayes.py TD+EMI kyoto3_TD_EMI KYOTO3
python3 naive_bayes.py PWC+TD+EMI kyoto3_PWC_TD_EMI KYOTO3

echo ARUBA
python3 naive_bayes.py BASE arubabase ARUBA
python3 naive_bayes.py PWC arubaclass ARUBA
python3 naive_bayes.py TD arubatime ARUBA
python3 naive_bayes.py MI arubami ARUBA
python3 naive_bayes.py EMI arubaEMI ARUBA
python3 naive_bayes.py PWC+TD arubaPWC_TD ARUBA
python3 naive_bayes.py PWC+EMI arubaPWC_EMI ARUBA
python3 naive_bayes.py TD+EMI arubaTD_EMI ARUBA
python3 naive_bayes.py PWC+TD+EMI arubaPWC_TD_EMI ARUBA