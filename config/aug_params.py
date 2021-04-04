class AugmentationParameters:
    def __init__(self, ni=None, ps=None):
        self.noise_injection = ni
        self.pitch_shift = ps
    
    def __str__(self):
        return str( dict( noise_injection = self.noise_injection, pitch_shift = self.pitch_shift ) )
    
    def set_chosen_transform(self, t):
        self.transform_chosen = t
    
    def samples_generated(self):
        ''' Returns dict containing current setting's generation number '''
        NOISE_INJECTION_STEPS = ((self.noise_injection[1] - self.noise_injection[0]) / self.noise_injection[2])
        PITCH_SHIFT_STEPS = ((self.pitch_shift[1] - self.pitch_shift[0]) / self.pitch_shift[2])
        return dict(
            ni_samples=NOISE_INJECTION_STEPS,
            ps_samples=PITCH_SHIFT_STEPS
        )
    
    def get_options_of_chosen_transform(self):
        if self.transform_chosen == 'ni':
            return self.noise_injection
        else:
            return self.pitch_shift