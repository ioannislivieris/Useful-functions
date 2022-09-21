def init_logger(log_file = 'logs.log'):
    
    from logging import getLogger
    from logging import INFO, ERROR, WARNING
    from logging import FileHandler
    from logging import Formatter
    from logging import StreamHandler


    logger = getLogger()
    logger.setLevel(INFO)
    
    
    
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("[%(levelname)s] %(message)s"))
    
    handler2 = FileHandler(filename = log_file)
    handler2.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%m/%d/%Y %I:%M:%S'))
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    
    return logger


# # Initiate logger
# #
# if VERBOSE:
#     logger = init_logger( log_file = 'logs.log' ) 

# if (VERBOSE):
#     logger.info('Info message')
#     logger.error('Error message')
#     logger.warning('Warning message')