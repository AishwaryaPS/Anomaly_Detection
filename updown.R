library(readr)
library(sampling)
library(dplyr)
library(moments)
library(caret)

kdd_10_rd <- read_csv("Desktop/5th Sem Project/Data Analytics/kdd_10_rd.csv")

kdd_10_rd$label <- ifelse(kdd_10_rd$result %in% c("ftp_write.", "guess_passwd.", "imap.", "multihop.", "phf.", "spy.", "warezclient.", "warezmaster."),"rtl",
                          ifelse(kdd_10_rd$result %in% c("back.","land.","neptune.","pod.","smurf.","teardrop."),"dos",
                                 ifelse(kdd_10_rd$result %in% c("ipsweep.","nmap.","portsweep.","satan."),"probe",
                                        ifelse(kdd_10_rd$result %in% c("buffer_overflow.","loadmodule.","perl.","rootkit."),"u2r","normal"))))#apply(kdd_10_rd, 2, attack_label)

table(kdd_10_rd$label)

dos = kdd_10_rd[(kdd_10_rd$label=="dos"),]
nor = kdd_10_rd[(kdd_10_rd$label=="normal"),]
oth = kdd_10_rd[!(kdd_10_rd$label=="dos" |kdd_10_rd$label=="normal"),]
n_indexes <- strata(arrange(nor,result), stratanames="result", size=0.45*c(table(nor$result)), method="srswor")
d_indexes <- strata(arrange(dos,result), stratanames="result", size=0.5*c(table(dos$result)), method="srswor")

drops <- c("Class")
u2r = oth[(oth$label=="u2r"),]
oth = oth[!(oth$label=="u2r"),]
table(u2r$result)
u2r$result <- as.factor(u2r$result)
up_u2r <- upSample(x = u2r, y = u2r$result)
up_u2r <- up_u2r[ , !(names(up_u2r) %in% drops)]
u_indexes <- strata(arrange(up_u2r,result), stratanames="result", size=c(30,22,12,22), method="srswor")

strat_sample_u <- up_u2r[u_indexes$ID_unit,]
strat_sample <- nor[n_indexes$ID_unit,]
strat_sample_d <- dos[d_indexes$ID_unit,]

table(dos$result)
table(strat_sample_d$result)
new <- rbind(strat_sample, oth)
new <- rbind(strat_sample_d, new)
new <- rbind(strat_sample_u, new)

table(new$label)

write.table(new,"Desktop/5th Sem Project/Data Analytics/Final3.csv",row.names=FALSE, na="", sep = ",")
