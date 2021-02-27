import numpy as np
import pandas as pd

from exctract_seqs_by_domain_architecture import get_target_query_names

target_domain_ids = ["PF001", "CL002"]
success = [["k1", "PF001", np.nan], ["k1", "PF002", "CL002"]]
fail1 = [["k2", "PF001", np.nan]]  # too less domains
fail2 = [["k3", "PF002", "CL002"], ["k3", "PF001", np.nan]]  # wrong order
expected = ["k1"]

columns = ["query_name", "pfam_id", "clan_id"]
data = success + fail1 + fail2
actual = get_target_query_names(
    pd.DataFrame(data, columns=columns), target_domain_ids
)
assert actual == expected, f"expected={expected}, actual={actual}"
print("OK.")
