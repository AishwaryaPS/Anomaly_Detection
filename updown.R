library(readr)
library(caret)
kdd_10_rd <- read_csv("kdd_10_rd.csv")

table(kdd_10_rd$result)

kdd_10_rd$label <- ifelse(kdd_10_rd$result %in% c("ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."),"rtl",
                          ifelse(kdd_10_rd$result %in% c("back.","land.","neptune.","pod.","smurf.","teardrop."),"dos",
                                 ifelse(kdd_10_rd$result %in% c("ipsweep.","nmap.","portsweep.","satan."),"probe",
                                        ifelse(kdd_10_rd$result %in% c("buffer_overflow.","loadmodule.","perl.","rootkit."),"u2r","normal"))))#apply(kdd_10_rd, 2, attack_label)
kdd_10_rd$label <- as.factor(kdd_10_rd$label)
table(kdd_10_rd$label)

set.seed(1)
down_kdd <- downSample(x = kdd_10_rd, y = kdd_10_rd$label)
table(down_kdd$label)
up_kdd <- upSample(x = kdd_10_rd, y = kdd_10_rd$label)
table(up_kdd$label)
