methods = ['gd', 'cd', 'ssd', 'spsa', 'rgfm', 'ssd_hf', 'ssd_bf', 'ssd_sag']

names = {
    'gd': 'GD',
    'cd': 'CD',
    'ssd': 'FS-SSD',
    'spsa': 'SPSA',
    'rgfm': 'GS',
    'ssd_hf': 'HF-SSD',
    'ssd_bf': 'BF-SSD',
    'ssd_oracle': 'O-SSD ',
    'ssd_sag': 'VR-SSD',
    'ssd_py': 'Polyak-SSD',
}

colors = {
    'gd': 'tab:blue',
    'cd': 'tab:orange',
    'ssd': 'tab:green',
    'spsa': 'tab:purple',
    'rgfm': 'tab:brown',
    'ssd_hf': 'tab:olive',
    'ssd_bf': 'tab:cyan',
    'ssd_oracle': 'tab:pink',
    'ssd_sag': 'tab:gray',
    'ssd_py': 'tab:red',
}