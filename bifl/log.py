from logging import getLogger, Formatter, StreamHandler, DEBUG

def setup_logging():
    lg = getLogger('malc')
    lg.setLevel(DEBUG)
    lh = StreamHandler()
    lh.setFormatter(Formatter('%(asctime)s - %(name)s %(process)-6s %(levelname)-3s - %(message)s'))
    lh.setLevel(DEBUG)
    lg.addHandler(lh)

def info(*_):
    lg = getLogger('malc').info(*_)

def warn(*_):
    lg = getLogger('malc').warn(*_)

def debug(*_):
    lg = getLogger('malc').debug(*_)
