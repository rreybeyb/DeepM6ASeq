args <- commandArgs(TRUE)
in_fn=args[1]
out_name=args[2]

library(data.table)
library(ggplot2)


x<-readLines(in_fn)

pdf(paste(out_name,".pdf",sep=""),width=8,height=4) 
m=0
for (i in 1:length(x) ) {
if (i==m*4+1) {
#print(x[i])
m=m+1
line_header=x[i]
line_prob=x[i+1]
line_nt=x[i+2]
line_scores=x[i+3]
prob=round(as.numeric(unlist(strsplit(line_prob, ":"))[2]),3)
nt=unlist(strsplit(line_nt, "\t"))
scores=unlist(strsplit(line_scores, "\t"))

df =data.frame(nt,scores)
df$scores=as.numeric(as.character(df$scores))
df$pos=seq(1:dim(df)[1])
df$pos=as.factor(df$pos)
df$dim="dim1"

prob_text=paste("prediction score: ",prob,sep="")
xlab_text=paste(prob_text,line_header,sep="\n")
p1=ggplot(df, aes(pos, dim)) + 
  geom_tile(aes(fill = scores))+scale_fill_gradient(low = "white",high="red")+
  scale_x_discrete(labels=df$nt)+theme(legend.position="top",axis.text.x=element_text(size=8),axis.text.y=element_blank(),axis.ticks.y =element_blank(),legend.text=element_text(size=5))+
  ylab("Saliency Map")+guides(fill = guide_legend(title = "saliency score"))+xlab(xlab_text)
print(p1)
}
}

dev.off()

