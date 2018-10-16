tcp<-subset(kdd_10_rd,protocol_type=="tcp")
udp<-subset(kdd_10_rd,protocol_type=="udp")
icmp<-subset(kdd_10_rd,protocol_type=="icmp")
boxplot(tcp[["srv_count"]],udp[["srv_count"]],icmp[["srv_count"]],names=c("TCP","UDP","ICMP"),xlab="SRV_COUNT", ylab="Protocol", main="Box plots of srv_count",horizontal=TRUE)



tcp_up<-subset(up_kdd,protocol_type=="tcp")
udp_up<-subset(up_kdd,protocol_type=="udp")
icmp_up<-subset(up_kdd,protocol_type=="icmp")
boxplot(tcp_up[["srv_count"]],udp_up[["srv_count"]],icmp_up[["srv_count"]],names=c("TCP","UDP","ICMP"),xlab="SRV_COUNT", ylab="Protocol",main="Box plots of srv_count",horizontal=TRUE)



icmp_outlier<-subset(icmp,srv_count>150)
tcp_outlier<-subset(tcp,srv_count>25)
udp_outlier<-subset(udp,srv_count>25)

icmp_outlier_up<-subset(icmp_up,srv_count>150)
tcp_outlier_up<-subset(tcp_up,srv_count>25)
udp_outlier_up<-subset(udp_up,srv_count>25)



table(icmp_outlier[["label"]])
icmp_dos<-subset(icmp,label=="dos")
icmp_normal<-subset(icmp,label=="normal")
icmp_probe<-subset(icmp,label=="probe")
icmp_rtl<-subset(icmp,label=="rtl")
icmp_u2r<-subset(icmp,label=="u2r")
boxplot(icmp_normal[["srv_count"]],icmp_probe[["srv_count"]],icmp_dos[["srv_count"]],icmp_rtl[["srv_count"]],icmp_u2r[["srv_count"]],names=c("Normal","Probe","DOS","RtL","U2R"),xlab="SRV_COUNT", main="Box plots of srv_count of classes of ICMP",horizontal=TRUE)

table(icmp_outlier_up[["label"]])
icmp_dos_up<-subset(icmp_up,label=="dos")
icmp_normal_up<-subset(icmp_up,label=="normal")
icmp_probe_up<-subset(icmp_up,label=="probe")
icmp_rtl_up<-subset(icmp_up,label=="rtl")
icmp_u2r_up<-subset(icmp_up,label=="u2r")
boxplot(icmp_normal_up[["srv_count"]],icmp_probe_up[["srv_count"]],icmp_dos_up[["srv_count"]],icmp_rtl_up[["srv_count"]],icmp_u2r_up[["srv_count"]],names=c("Normal","Probe","DOS","RtL","U2R"),xlab="SRV_COUNT", main="Box plots of srv_count of classes of ICMP",horizontal=TRUE)





#row_prob_dist <- function(row) {
#  return (t(lapply(row, function(x,y=sum(row)) x/y)))
#}
freq_dist<-table(kdd_10_rd[["protocol_type"]],kdd_10_rd[["label"]])

freq_dist_up<-table(up_kdd[["protocol_type"]],up_kdd[["label"]])

#data.frame(freq_dist)

barchart(freq_dist,col=1:5, horizontal=FALSE, xlab="Protocol",ylab="Frequency of a type of attack")

barchart(freq_dist_up,col=1:5, horizontal=FALSE, xlab="Protocol",ylab="Frequency of a type of attack")

library("corrplot")
numeric_kdd<- unlist(lapply(kdd_10_rd, is.numeric)) 
corrplot(cor(kdd_10_rd[,numeric_kdd]))


numeric_kdd_up<- unlist(lapply(up_kdd, is.numeric)) 
corrplot(cor(up_kdd[,numeric_kdd_up]))
