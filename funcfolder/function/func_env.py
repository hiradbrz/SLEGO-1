# a function install python package
def install(package:str = 'pandas'):
    # use pip to install package
    # pip install package
    import pip
    pip.main(['install', package])