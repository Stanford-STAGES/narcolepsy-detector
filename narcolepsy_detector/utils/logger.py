import os
import sys
import logging


def get_logger(save_dir=None):
    logger = logging.getLogger("narcolepsy-detector")

    if logger.handlers:
        return logger

    # Add handlers
    shell_handler = logging.StreamHandler()
    if save_dir is not None:
        file_handler = logging.FileHandler(save_dir)

    # Set output levels
    logger.setLevel(logging.INFO)
    shell_handler.setLevel(logging.INFO)
    if save_dir is not None:
        file_handler.setLevel(logging.INFO)

    # Set formatter
    fmt_shell = "%(asctime)s | %(levelname)s | %(message)s"
    if save_dir is not None:
        fmt_file = "%(asctime)s | %(levelname)s | [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s"

    # Attach the formatters
    shell_formatter = logging.Formatter(fmt_shell, datefmt="%H:%M:%S")
    if save_dir is not None:
        file_formatter = logging.Formatter(fmt_file, datefmt="%H:%M:%S")

    # Hook everything
    shell_handler.setFormatter(shell_formatter)
    if save_dir is not None:
        file_handler.setFormatter(file_formatter)
    logger.addHandler(shell_handler)
    if save_dir is not None:
        logger.addHandler(file_handler)

    return logger


def add_file_handler(logger, save_dir, filename='model.log'):
    fh = logging.FileHandler(os.path.join(save_dir, filename))
    fh.setLevel(logging.INFO)
    assert logger.handlers, "Logger object has no handlers!"
    fh.setFormatter(logger.handlers[0].formatter)  # should already have a stream handler
    logger.addHandler(fh)
    return logger


def remove_file_handler(logger):
    [logger.removeHandler(hdlr) for hdlr in logger.handlers if isinstance(hdlr, logging.FileHandler)]
    return logger


def print_args(logger, args):

    logger.info(f'Usage: {" ".join([x for x in sys.argv])}\n')
    logger.info("Settings:")
    logger.info("---------")
    for idx, (k, v) in enumerate(sorted(vars(args).items())):
        if idx == len(vars(args)) - 1:
            logger.info(f"{k:>15}\t{v}\n")
        else:
            logger.info(f"{k:>15}\t{v}")
