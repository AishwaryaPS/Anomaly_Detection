library(plyr)
#data <- read.csv(file="/Users/aishwarya/Downloads/kddcup.csv", header=FALSE, sep=",");
cat("There are only 3 protocol types in this dataset\n")
print(unique(kdd_rd$protocol_type),max.levels = 0)
protocol <- count(kdd,c('V2'))
pie(protocol$freq,
    labels=as.character(protocol$V2),
    main="Original Protocol type distribution")
protocol <- count(kdd_rd,c('protocol_type'))
pie(protocol$freq,
    labels=as.character(protocol$protocol_type),
    main="Protocol type for removed duplicates")
protocol <- count(up_kdd,c('protocol_type'))
pie(protocol$freq,
    labels=as.character(protocol$protocol_type),
    main="Protocol type for upsampling")
cat("As seen from the three pie charts, Most of the icmp connetions are redundant and have been removed from the kdd_rd dataset. \nThis is probably because the icmp connections might be simple ping messages that tend to repeat and hence lead to redundant rows.On removal of duplicate rows, we can see how much the proportion of icmp reduces.\n")
service1 <- count(kdd_rd,c('service'))
service2 <- count(up_kdd,c('service'))
service <- cbind(service1,service2$freq)
colnames(service) <- c('service','no_dup_freq','up_freq')
service <- service[order(-service$no_dup_freq),]
cat("There are 66 services. These are the network services running on the destination\n")
print(service)
boxplot(count ~ label, data = kdd_10_rd, xlab = "Attack domains",
        ylab = "No. of connections", main = "Same host connections in past 2s")
cat("From the boxplot we can see that DOS has a large number of connections with the host, even on average. So it always requires alot of connections with the host. Probing attacks can vary, since you can make the connections frequently or you can range it out,like 1 every 2 minutes.\nrtl and u2r attacks require 1 or a few connections for the attack usually as can be seen.\n")
numeric_kdd_rd <- kdd_rd[, !(colnames(kdd_rd) %in% c('protocol_type','service','flag','result'))]
cov <- cov(numeric_kdd_rd)
host_login <- var(kdd_rd$is_host_login)
num_outbound_cmds <- var(kdd_rd$num_outbound_cmds)
cat("The covariance matrix shows two attributes(is_host_login, num_outbound_cmds) having 0 covariance with every other attribute.\n")
cat("The variance of is_host_login is ",host_login," which means it can definitely be dropped\n")
cat("The variance of num_outbound_cmds is ",num_outbound_cmds," which means it can definitely be dropped\n")
kdd_rd <- kdd_rd[, !(colnames(kdd_rd) %in% c('is_host_login','num_outbound_cmds'))]
down_kdd <- down_kdd[, !(colnames(down_kdd) %in% c('is_host_login','num_outbound_cmds'))]
up_kdd <- up_kdd[, !(colnames(up_kdd) %in% c('is_host_login','num_outbound_cmds'))]
