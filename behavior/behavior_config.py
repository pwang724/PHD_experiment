class behaviorConfig(object):
    def __init__(self):
        #extra time (seconds) given to CS- odors after water onset for quantification.
        self.extra_csm_time = 0
        # self.smoothing_window = 11
        # self.smoothing_window_boolean = 11
        self.polynomial_degree = 0

        self.smoothing_window_first_lick = 7
        self.rules_output_lick = {'PT CS+': 19}
        self.rules_output_boolean = {'PT CS+': 19}
        self.rules_single_phase_lick = {'CS+': 19, 'CS-': 19}
        self.rules_single_phase_boolean = {'CS+': 19, 'CS-': 19}
        self.rules_two_phase_lick = {'PT CS+': 19, 'PT Naive':19, 'CS+': 11, 'CS-': 11}
        self.rules_two_phase_boolean = {'PT CS+': 19, 'PT Naive':19, 'CS+': 5, 'CS-': 11}

        self.halfmax_up_threshold = 50
        self.halfmax_down_threshold = 50

        self.fully_learned_threshold_up = 80
        self.fully_learned_threshold_down = 20

        #smoothing window = 11 for head-fixed mush behavior (fig 1)
        #smoothing window = 19 for head-fixed mush inhibition / behavior
        #smoothing window = 11 for head-fixed two-phase pre-training inhibition / behavior
        #smoothing window = 5 for head-fixed two-phase discrimination inhibition / behavior

        # 3 for ofc reversal
        # 1 for ofc state