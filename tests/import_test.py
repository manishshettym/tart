from tart.utils.model_utils import get_device


def import_test():
    device = get_device()
    print(device)
    return True
