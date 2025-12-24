from g2p_en import G2p
from phonecodes import phonecodes
import nltk

nltk.download("averaged_perceptron_tagger")

g2p = G2p()


def convert_to_ipa(word):
    arpabet = g2p(word)
    arpabet = " ".join(arpabet)
    ipa = phonecodes.convert(arpabet, "arpabet", "ipa", "eng")
    ipa = ipa.replace(" ", "").replace("ˌ", "").replace("ˈ", "")
    return ipa