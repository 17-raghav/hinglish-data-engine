import pytest
from hde.numbers_hi import to_words, parse_numbers, verbalize_numbers, digits_to_words
from hde.entities import parse_numbers_en, check_pair


@pytest.mark.parametrize("n", [0, 7, 19, 45, 99, 100, 101, 250, 999, 1000, 1500, 25000,
                               100000, 250000, 1234567, 10000000, 350000000, 999999999])
def test_round_trip(n):
    assert parse_numbers(to_words(n)) == [float(n)]


@pytest.mark.parametrize("text,value", [
    ("ढाई लाख रुपये", 250000), ("डेढ़ हज़ार", 1500), ("साढ़े तीन लाख", 350000),
    ("सवा सौ", 125), ("पौने दो सौ", 175), ("साढ़े सात सौ", 750),
    ("दो लाख पचास हज़ार", 250000), ("दस करोड़", 10**8), ("तीन दशमलव पाँच", 3.5),
])
def test_hindi_fractions_and_scales(text, value):
    assert parse_numbers(text) == [value]


def test_digit_string():
    assert parse_numbers("खाता नंबर नौ आठ सात छह है") == [9876.0]
    assert digits_to_words("4021") == "चार शून्य दो एक"


def test_verbalize():
    assert verbalize_numbers("₹2,50,000 भेजो") == "दो लाख पचास हज़ार रुपये भेजो"
    assert verbalize_numbers("मेरा नंबर 9876543210 है") == "मेरा नंबर नौ आठ सात छह पाँच चार तीन दो एक शून्य है"
    assert verbalize_numbers("12% ब्याज") == "बारह प्रतिशत ब्याज"


@pytest.mark.parametrize("text,values", [
    ("I sent 2.5 lakh rupees", [250000]), ("two lakh fifty thousand", [250000]),
    ("Rs 2,50,000 and 25k", [250000, 25000]), ("account nine eight seven six", [9876]),
    ("three hundred and fifty", [350]), ("one and a half lakh", [150000]),
])
def test_english_numbers(text, values):
    assert parse_numbers_en(text) == values


def test_check_pair_catches_wrong_amount():
    hi = "मैंने ढाई लाख रुपये भेजे।"
    assert check_pair(hi, "I sent 2.5 lakh rupees.") == []
    assert check_pair(hi, "I sent 2 lakh rupees.") != []


def test_check_pair_self_correction():
    hi = "तीन लाख, नहीं नहीं, साढ़े तीन लाख भेजो।"
    # translator declares 300000 as abandoned
    assert check_pair(hi, "Send three and a half lakh, no wait.", dropped=[300000]) == []


def test_negation():
    assert check_pair("मैंने खाता बंद नहीं किया।", "I closed the account.") != []
    assert check_pair("मैंने खाता बंद नहीं किया।", "I did not close the account.") == []


from hde.names import name_in_english, name_score


def test_names_survive_script_change():
    for hi, en in [("हरप्रीत सिंह", "Harpreet Singh paid"), ("पंचकूला", "the Panchkula branch"),
                   ("मोहम्मद आसिफ़", "Mohammed Asif called"), ("लखनऊ", "in Lucknow"),
                   ("देहरादून", "in Dehradun")]:
        assert name_in_english(hi, en), (hi, en, name_score(hi, en))


def test_wrong_names_are_caught():
    for hi, en in [("रमेश शर्मा", "Rakesh Verma"), ("पंचकूला", "the Pune branch"),
                   ("कविता नायर", "Kavya Nayak")]:
        assert not name_in_english(hi, en)


def test_check_pair_names():
    hi = "रमेश शर्मा ने पैसे नहीं भेजे।"
    assert check_pair(hi, "Ramesh Sharma did not send the money.", names=["रमेश शर्मा"]) == []
    assert check_pair(hi, "Rakesh did not send the money.", names=["रमेश शर्मा"]) != []


def test_precomposed_nukta_forms_parse_the_same():
    assert parse_numbers("पचास हज़ार") == [50000.0]          # precomposed ज़
    assert parse_numbers("डेढ़ लाख") == [150000.0]    # precomposed ढ़
