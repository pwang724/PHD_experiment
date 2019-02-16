from collections import namedtuple

#CSP are always the first two odors

def all_conditions():
    out = [
        PIR, PIR_NAIVE,
        OFC, OFC_COMPOSITE, OFC_LONGTERM, OFC_STATE, OFC_CONTEXT, OFC_REVERSAL,
        BLA, BLA_LONGTERM, BLA_STATE, BLA_CONTEXT, BLA_REVERSAL,
        OFC_JAWS, BLA_JAWS,
        MPFC_COMPOSITE
    ]
    return out

class PIR:
    name = 'PIR'
    paths = ['E:/IMPORTANT DATA/DATA_2P/M183/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M184/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M199_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M200_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M201_pir/training_LEARNING',
             'E:/IMPORTANT DATA/DATA_2P/M202_pir/training_LEARNING']
    odors = [
        ['pin', 'msy', '4mt', 'lim'],
        ['pin', 'msy', '4mt', 'lim'],
        ['euy', 'lim', 'iso', '4mt'], #pin
        ['euy', 'lim', 'iso', '4mt'], #pin
        ['euy', 'lim', 'fen', 'ger'],
        ['euy', 'lim', 'iso', '4mt'], #pin
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['euy', 'lim'],
           ['euy', 'lim'],
           ['euy', 'lim'],
           ['euy', 'lim']
           ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0, 0]
    min_neurons = 46

class PIR_NAIVE:
    name = 'PIR_NAIVE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M199_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M200_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M201_pir/training_NAIVE',
        'E:/IMPORTANT DATA/DATA_2P/M202_pir/training_NAIVE'
             ]
    odors = [
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct'],
        ['2pe', 'ben', 'msy', 'oct']
    ]
    timing_override = [True, True, True, True]
    min_neurons = 46


class OFC:
    name = 'OFC'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_LEARNING'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [False, False, False, True, True]
    training_start_day = [1, 1, 1, 1, 1]
    min_neurons= 52

class OFC_STATE:
    name = 'OFC_STATE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_STATE'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0]

class OFC_CONTEXT:
    name = 'OFC_CONTEXT'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_CONTEXT'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['euy', 'lim'],
        ['euy', 'lim'],
        ['euy', 'lim'],
        ['euy', 'lim']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]

class OFC_REVERSAL:
    name = 'OFC_REVERSAL'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M187_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M188_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M206_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M233_ofc/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M234_ofc/training_REVERSAL'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0]

class OFC_LONGTERM:
    name = 'OFC_LONGTERM'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M230_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M232_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M239_ofc/training_LONGTERM',
        'E:/IMPORTANT DATA/DATA_2P/M241_ofc/training_LONGTERM'
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]
    min_neurons = 58

class OFC_JAWS:
    name = 'OFC_JAWS'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M222_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M226_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M227_jawsofc_bla/training_LEARNING',
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0]

class BEHAVIOR_OFC_JAWS_MUSH:
    name = 'BEHAVIOR_OFC_JAWS_MUSH'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M222_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M226_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M227_jawsofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M221_jawsofc_bla',
    ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]

class OFC_COMPOSITE:
    name = 'OFC_COMPOSITE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M2_OFC/training', #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M3_OFC/training', #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M4_OFC/training', #days 1-6
        'E:/IMPORTANT DATA/DATA_2P/M5_OFC/training', #days 1-4
             ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]
    pt_odors = [
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct'],
        ['naive', 'oct']
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]
    timing_override = [True, True, True, True]
    training_start_day = [1, 1, 1, 1]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [1, 1, 1, 1]
    last_pt_day = [3, 3, 5, 3]
    first_dt_day = [4, 4, 6, 4]


class MPFC_COMPOSITE:
    name = 'MPFC_COMPOSITE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M2_MPFC/training',   #days 1-3
        'E:/IMPORTANT DATA/DATA_2P/M3_MPFC/training',   #days 1-3
        'E:/IMPORTANT DATA/DATA_2P/M4_MPFC/training',   #days 1-4
        'E:/IMPORTANT DATA/DATA_2P/M6_MPFC/training'    #days 1-4
    ]
    dt_odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]

    dt_csp = [
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
        ['pin', 'msy'],
    ]

    pt_odors = [
        ['oct'],
        ['oct'],
        ['naive','oct'],
        ['naive','oct'],
    ]
    pt_csp = [
        ['oct'],
        ['oct'],
        ['oct'],
        ['oct']
    ]
    timing_override = [True, True, True, True]
    training_start_day = [1, 1, 1, 1]
    naive_dt_day = [0, 0, 0, 0]
    naive_pt_day = [None, None, 1, 1]
    last_pt_day = [2, 2, 3, 3]
    first_dt_day = [3, 3, 4, 4]
    min_neurons = 49

class BLA_JAWS:
    #TODO
    name = 'BLA_JAWS'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M220_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M223_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M224_jawsbla_ofc/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M225_jawsbla_ofc/training_LEARNING'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]

class BLA:
    name = 'BLA'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M213_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M214_bla/training_LEARNING',
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy']
           ]
    timing_override = [True, True, True, True]
    training_start_day = [1, 1, 1, 1]
    min_neurons = 22

class BLA_STATE:
    name = 'BLA_STATE'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_STATE',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_STATE'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['euy', 'lim'],
        ['euy', 'lim']
           ]
    timing_override = [True, True]

class BLA_CONTEXT:
    name = 'BLA_CONTEXT'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_CONTEXT',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_CONTEXT'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin','msy'],
        ['pin','msy']
           ]
    timing_override = [True, True]

class BLA_REVERSAL:
    name = 'BLA_REVERSAL'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_REVERSAL',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_REVERSAL'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [
        ['pin','msy'],
        ['pin','msy']
           ]
    timing_override = [True, True]
class BLA_LONGTERM:
    name = 'BLA_LONGTERM'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/M211_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M212_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M213_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M214_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M260_ofc_bla/training_LEARNING',
        'E:/IMPORTANT DATA/DATA_2P/M261_bla/training_LEARNING'
        ]
    odors = [
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim'],
        ['pin', 'msy', 'euy', 'lim']
    ]
    csp = [['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy'],
           ['pin', 'msy']
           ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [1, 1, 1, 1, 1, 1]

class BEHAVIOR_OFC_YFP:
    name = 'BEHAVIOR_OFC_YFP'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/C1_M247',
        'E:/IMPORTANT DATA/DATA_2P/C2_M248',
        'E:/IMPORTANT DATA/DATA_2P/C3_M249',
        'E:/IMPORTANT DATA/DATA_2P/C4_M250',
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]
    last_pt_day = [2, 2, 2, 2]

class BEHAVIOR_OFC_JAWS_PRETRAINING:
    name = 'BEHAVIOR_OFC_JAWS_PRETRAINING'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/J3_M245',
        'E:/IMPORTANT DATA/DATA_2P/J5_M251',
        'E:/IMPORTANT DATA/DATA_2P/J6_M252',
        'E:/IMPORTANT DATA/DATA_2P/J7_M253',
        'E:/IMPORTANT DATA/DATA_2P/J8_M262',
        'E:/IMPORTANT DATA/DATA_2P/J9_M263',
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    timing_override = [True, True, True, True, True, True]
    training_start_day = [0, 0, 0, 0, 0, 0]
    last_pt_day = [4, 3, 3, 1, 2, 2]

class BEHAVIOR_OFC_JAWS_DISCRIMINATION:
    name = 'BEHAVIOR_OFC_JAWS_DISCRIMINATION'
    paths = [
        'E:/IMPORTANT DATA/DATA_2P/JD10_256',
        'E:/IMPORTANT DATA/DATA_2P/JD11_257',
        'E:/IMPORTANT DATA/DATA_2P/JD12_258',
        'E:/IMPORTANT DATA/DATA_2P/JD13_259',
    ]
    dt_odors = [
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
        ['pin', 'fen', 'euy', 'lim'],
    ]
    dt_csp = [
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
        ['pin', 'fen'],
    ]
    pt_odors = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    pt_csp = [
        ['msy'],
        ['msy'],
        ['msy'],
        ['msy'],
    ]
    timing_override = [True, True, True, True]
    training_start_day = [0, 0, 0, 0]
    last_pt_day = [3, 3, 2, 2]



