def check_edge_type(flag1,flag2):
    if flag1 and flag2:
        # Internal transaction
        return 0
    elif flag1 and not flag2:
        # Internal -> External
        return 1
    elif not flag1 and flag2:
        # External -> Internal
        return 2
    else:
        # External -> External, which should be deleted as we don't have the information
        return -1