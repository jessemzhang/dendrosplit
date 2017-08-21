
#########################################################
# This program is part of the SNN-Cliq method           #
# Contact Chen Xu at UNC-Charlotte for more information.# 
#########################################################

# 6/14/16
# Modified by Jesse Zhang at Cellular Research so that you can run this 
# file from the command line prompt, e.g.:
#
# Rscript SNN.R DISTFILE EDGEFILE KVAL
#
# Input to the SNN function has also been changed from the data matrix to 
# a distance matrix.

args<-commandArgs(TRUE)

SNN<-function(x, outfile, k){

	if(missing(x)){
		stop(paste("Input distance matrix missing.",help,sep="\n"))
	}
	if(missing(outfile)){
		stop(paste("Output file name missing.",help,sep="\n"))
	}
	if(missing(k)){
		k=3
	}
  numSpl<-dim(x)[1]
	IDX<-t(apply(x,1,order)[1:k,]) # knn list

	edges<-matrix(0,numSpl^2,3)              # SNN graph
	edge_ind<-1
	for (i in 1:numSpl){
		j<-i
		while (j<numSpl){
			j<-j+1
			# find how many neighbors a point shares with each other point
			shared<-intersect(IDX[i,], IDX[j,])
			if(length(shared)>0){			
				s<-k-0.5*(match(shared, IDX[i,])+match(shared, IDX[j,]))
				strength<-max(s)
				if (strength>0)
					edges[edge_ind,]<-c(i,j,strength)
				  edge_ind<-edge_ind+1
			}				
		}
	}
	edges<-head(edges,edge_ind-1)
	write.table(edges, outfile, quote=FALSE, sep='\t',col.names=FALSE,row.names=FALSE)
}

distfile<-args[1]
outfile<-args[2]
kval<-strtoi(args[3])

x<-as.matrix(read.table(distfile));
SNN(x, outfile, k=kval)
