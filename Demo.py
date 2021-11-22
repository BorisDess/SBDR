import numpy as np
import pandas as pd
import datetime
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import os 

os.chdir('/Users/boris/Nextcloud/OBIWAN/DATASETS')
# os.chdir('C:/Users/Adrian/Nextcloud/OBIWAN/DATASETS')

global ListMedianCenter

dfCheckMissingSensors = pd.DataFrame()
def PrepareDataSet(DataSetName):
    DataSet = pd.read_csv(DataSetName + ".csv", index_col = 0)
    DataSet["date"] = DataSet.index
    DataSet["date"] = DataSet["date"].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
    DataSet = DataSet.loc[:, ["date", "pm10", "node_id"]]
    TablePivot = pd.pivot_table(DataSet, index=["date"], columns="node_id", values = "pm10")
    
    for Col in TablePivot.columns :
        dfCheckMissingSensors.loc[DataSetName, Col] = len(TablePivot[Col].dropna())
       
    for ColToDrop in [7739291, 25664263, 25464263, 67054918, 72634847] :
        if ColToDrop in TablePivot.columns : 
            TablePivot = TablePivot.drop(columns = ColToDrop)
    
    Columns = [ 1664163, 11313650, 12961312, 22387298, 23528066,
            25610140, 25817166, 27753754, 34264592, 34513136, 36737469,
            41118302, 41162841, 49656412, 51834402, 53612054,
            53803260, 53857043, 54046833, 58131916, 58400885, 62938333,
            63713358, 65591750, 83580945, 87458302]
    
    TablePivot = TablePivot[Columns]
    
    print(DataSetName + " sensors : ")
    print(TablePivot.columns)
    DataSet = TablePivot.values

    return(DataSet)

def SplitDataSet(DataSet) :
    # DataSet = DataSet2
    TrainingSplit = int(np.round(DataSet.shape[0]*SplitRatio))
    RandomIdx = np.random.permutation(np.arange(0, DataSet.shape[0]))
    TrainingIdx, TestIdx = RandomIdx[:TrainingSplit], RandomIdx[TrainingSplit:]
    SourceDataSet, SubjectDataset = pd.DataFrame(DataSet).iloc[TrainingIdx,:].values, pd.DataFrame(DataSet).iloc[TestIdx,:].values

    return(SourceDataSet, SubjectDataset, DataSet)

def CheckDs(DataSetToCheck, MinSensor):
    
    DataSetToCheck = pd.DataFrame(DataSetToCheck)
    ListLen = []
    for Col in DataSetToCheck.columns :
        Len = len(DataSetToCheck[Col].dropna())
        ListLen.append(Len)
        print(str(Col) + " " + str(Len))
    np.min(ListLen)

    DataSetToCheck["CountCol"] = np.nan
    for i in DataSetToCheck.index :
        ()
        Data = DataSetToCheck.loc[i,:]
        DataSetToCheck.loc[i,"CountCol"] = len(Data.dropna())
    DataSetToCheck = DataSetToCheck[DataSetToCheck.loc[:, "CountCol"]  >= MinSensor]
    DataSetToCheck.drop("CountCol", axis = 1, inplace = True)
    DataSetToCheck = DataSetToCheck.values
    return(DataSetToCheck, ListLen)

#%% Airparif Cairsens

SourceDataSet = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Cairsens_Airparif.csv', index_col = 0)
SourceDataSet.index = pd.to_datetime(SourceDataSet.index.values.astype(np.int64)*1000000000)

RealSourceDataSet = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Cairsens_Airparif_REF_SOURCE.csv', index_col = 0)
RealSourceDataSet.index = pd.to_datetime(RealSourceDataSet.date.values, format='%Y-%m-%d %H:%M:%S')
RealSource = RealSourceDataSet.resample('60s').mean()

#%%
        
DataSet = SourceDataSet.loc[RealSource.dropna().index,:]
DataSet.loc[:,"RealSource"] = RealSource.loc[RealSource.dropna().index,:].values
AirparifDates = DataSet.index

Airparif, ListLen = CheckDs(DataSet, 10)
# np.nanmean(np.nanstd(Airparif[:,:-1], axis = 1))


#%% LSCE Canarins
SourceDataSet = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Canarins_LSCE.csv', index_col = 0)
SourceDataSet.index = pd.to_datetime(SourceDataSet.index.values, format='%Y-%m-%d %H:%M:%S')
ResampledDataSet = SourceDataSet.resample('60s').mean()

Lsce, ListLen = CheckDs(ResampledDataSet, 6)



#%% Canarins Coimbra

SourceDataSet = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra.csv', index_col = 0, header = 4)
# A = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Juin.csv', index_col = 0, header = 4)
# B = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Juillet.csv', index_col = 0, header = 4)
# C = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Août.csv', index_col = 0, header = 4)
# D = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Septembre.csv', index_col = 0, header = 4)
# E = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Mars.csv', index_col = 0, header = 4)
# F = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Avril.csv', index_col = 0, header = 4)
# G = pd.read_csv('~/Nextcloud/DISPERSION/datasets/Coimbra2/Mai.csv', index_col = 0, header = 4)

# Coimbra = pd.concat([SourceDataSet, A, B, C, D, E, F, G])
# Coimbra.plot()
SourceDataSet = SourceDataSet[['Node', 'PM2.5']]

SourceDataSet.index = pd.to_datetime(SourceDataSet.index.values.astype(np.int64)*1000000000)
SourceDataSet = SourceDataSet.pivot(columns = 'Node', values = 'PM2.5')
SourceDataSet = SourceDataSet.resample('60s').mean()

# SourceDataSet.num11 = SourceDataSet.num11 + 10
# SourceDataSet.num12 = SourceDataSet.num12 + 20
# SourceDataSet.num13 = SourceDataSet.num13 + 30
# SourceDataSet.num14 = SourceDataSet.num14 + 40
SourceDataSet = SourceDataSet.iloc[10000:101000]
SourceDataSet.plot()

Coimbra, ListLen = CheckDs(SourceDataSet, 3)


#%% OBIWAN datasets

DataSet1 = PrepareDataSet("Test1Dataset")
DataSet2 = PrepareDataSet("Test2Dataset")
DataSet3 = PrepareDataSet("Test3Dataset")

DataSet1, ListLen = CheckDs(DataSet1, 5)
DataSet2, ListLen = CheckDs(DataSet2, 5)
DataSet3, ListLen = CheckDs(DataSet3, 5)

# SubjectDatasetStd = np.nanmean(np.std(DataSet2[:, :], axis = 1))



#%% PLOT DISPLAY REF VS CORRECTED

CorrectedDataSet_NoREF
CorrectedDataSet_WithREF = pd.read_excel("CorrectedDataSet_WithREF.xlsx", index_col = 0)

def RemoveCenter(DataSet):
    DataSet.index = AirparifDates
    DataSet.iloc[:,1:] = (DataSet.iloc[:,1:].values.T - DataSet.iloc[:,0].values).T
    DataSet = DataSet.iloc[:,1:]
    return(DataSet)

def MinMaxResample(DataSet):
    Max = DataSet.min(axis = 1).resample("1200s").max()
    Min = DataSet.min(axis = 1).resample("1200s").min()
    return(Min, Max)

MinRef, MaxRef = MinMaxResample(RemoveCenter(CorrectedDataSet_WithREF))
MinNoRef, MaxNoRef = MinMaxResample(RemoveCenter(CorrectedDataSet_WithREF))

# plt.plot(CorrectedDataSet_WithREF)


plt.plot(MinNoRef, color = "red", alpha = 0.2)
plt.plot(MaxNoRef, color = "red", alpha = 0.2)

plt.plot(MinRef, color = "grey", alpha = 0.2)
plt.plot(MaxRef, color = "grey", alpha = 0.2)



fig = plt.figure()
# Median = np.nanmedian(DataSet, axis = 1)

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], lw=4, color = "black"),
                Line2D([0], [0], lw=4, color = "orangered"),
                Line2D([0], [0], lw=4, color = "#6C41B0")]



# plt.plot(CorrectedDataSet_WithREF, color = "teal", alpha = 0.1)

# plt.plot(Airparif[:,:-1], color = "orangered", alpha = 1)  



# label = 
# label = 
# plt.plot(CorrectedDataSet[:,0:], alpha = 0.2, color = "blue")
# plt.xticks = np.arange(0, len(DataSet))

# plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
plt.legend()
plt.legend(custom_lines, ['Données brutes', 'Données corrigées', "Centres estimés"])
plt.ylabel("PM10 (µg/m³)")
plt.xlabel("Temps (minutes)")
plt.show()
    

#%% THESIS : Démonstration de l’estimation des centres
  
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 2 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
RoundEAbsolutes = True # EAbsolutes are rounded, thus merging all EAbsolutes together and averaging EValues
EXT_RatioCenters = 1

DataSetName = "Test2Dataset"
DataSet = pd.read_csv(DataSetName + ".csv", index_col = 0)
DataSet["date"] = DataSet.index
DataSet["date"] = DataSet["date"].apply(lambda x : datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
DataSet = DataSet.loc[:, ["date", "pm10", "node_id"]]
TablePivot = pd.pivot_table(DataSet, index=["date"], columns="node_id", values = "pm10")

DataSet = TablePivot.resample("60s").mean()

Columns = [ 1664163, 11313650, 12961312, 22387298, 23528066,
        25610140, 25817166, 27753754, 34264592, 34513136, 36737469,
        41118302, 41162841, 49656412, 51834402, 53612054,
        53803260, 53857043, 54046833, 58131916, 58400885, 62938333,
        63713358, 65591750, 83580945, 87458302]

DataSet = DataSet[Columns]

DataSet, ListLen = CheckDs(DataSet, 5)
# SourceDataSet, SubjectDataset, DataSet = SplitDataSet(DataSet)
EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(DataSet)
# EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, OffsetsSum = ApplyBDRCorrection(EValuesSource, DataSet)

# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], lw=4, color = "black", alpha = 0.2),
#                 Line2D([0], [0], lw=4, color = "teal")]


EXT_CenterFunction = "limit"
EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(DataSet)

fig = plt.figure()
# plt.legend(custom_lines, ['Capteurs', 'Centre'])
Median = np.nanmedian(DataSet, axis = 1)

plt.plot(DataSet[:,1:], color = "black", alpha = 0.2)
plt.plot(DataSet[:,0], color = "black", alpha = 0.2, label = "Capteurs")

n = 0
# ColorList = ["#FFF850", "#FC7784", "#5DC7FC"]
# for RatioCenters in np.arange(0.1, 0.8, 0.3) :
#     Color = ColorList[n]
#     n = n + 1
#     EXT_CenterFunction = "limit"
#     EXT_RatioCenters = np.round(RatioCenters, 1)
#     EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(DataSet)
#     # EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, OffsetsSum = ApplyBDRCorrection(EValuesSource, DataSet)
#     plt.plot(EAbsolutesSource, color = Color, alpha = 0.5, label = "Sélectivité " + str(int(EXT_RatioCenters*100)) + " %")
    
plt.plot(EAbsolutesSource, color = "#6C41B0", alpha = 1, label = "Critère Sélection 10%", lw = 3)

# plt.plot(CorrectedDataSet[:,0:], alpha = 0.2, color = "blue")
plt.xticks = np.arange(0, len(DataSet))

# plt.tick_params(
    # axis='x',          # changes apply to the x-axis
    # which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    # top=False,         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
# plt.legend()
plt.ylabel("PM10 (µg/m³)")
plt.xlabel("Temps (minutes)")

plt.show()


#%% TEST ALL VALUES X/Y

global EXT_RatioCenters
global EXT_FloorRatio
global EXT_CeilingRatio
global EXT_RatioCentersI 
global EXT_RatioCentersII 
global EXT_RatioCentersIII

EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
CorrectionDecPre = 0
SplitRatio= 0.4
GenerateControledDataSet = False
GenerateErrorPerAbsolute = False
PrintMissingEAbsolutes = False

AllIn = np.concatenate([DataSet1, DataSet2, DataSet3])

EXT_CenterFunction = "limit"

CeilingRatioRange = [0.1, 0.25, 0.5, 1, 2, 3]

FloorRatioRange = [0.05, 0.1, 0.25, 0.5]

RatioCenterRange = [0.1, 0.25, 0.50, 0.75]
RatioCenterRangeA, RatioCenterRangeB, RatioCenterRangeC = (RatioCenterRange, RatioCenterRange, RatioCenterRange)


ListDataSets = [(AllIn, DataSet1, "DataSet1"), (AllIn, DataSet2, "DataSet2"), (AllIn, DataSet3, "DataSet3")] # DONE

# ListDataSets = [(Lsce, Lsce, "Lsce"), (Airparif, Airparif, "Airparif"), (Coimbra, Coimbra, "Coimbra")]
TotalDuration = False

TotalCorrections = len(CeilingRatioRange)*len(FloorRatioRange)*len(RatioCenterRangeA)*len(RatioCenterRangeB)*len(RatioCenterRangeC)

for SourceDataSet, SubjectDataSet, Name in ListDataSets : 
    TimeStart = datetime.datetime.now()
    CountCorrections = 0

    ScoreDict = {}
    for CeilingRatio in CeilingRatioRange :
        EXT_CeilingRatio = CeilingRatio
        for FloorRatio in FloorRatioRange :
            EXT_FloorRatio = FloorRatio
            
            ScoreArray = np.full((len(RatioCenterRangeA),len(RatioCenterRangeB),len(RatioCenterRangeC)), np.nan)
            a = -1
            for RatioCentersA in RatioCenterRangeA :
                a = a + 1
                b = -1
                EXT_RatioCenters = RatioCentersA
                EAbsolutes = ComputeCenters(SourceDataSet)
                        
                for RatioCentersB in RatioCenterRangeB :
                    b = b + 1
                    c = -1
                    EXT_RatioCenters = RatioCentersB
                    EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataSet, EAbsolutes)

                    for RatioCentersC in RatioCenterRangeC :
                        c = c + 1
                        EXT_RatioCenters = RatioCentersC
                        CountCorrections = CountCorrections + 1                        
                        Now = datetime.datetime.now()
                             
                        # n = n+1
                        print("DataSet : " + Name + "\n"
                              "Ceiling : " + str(EXT_CeilingRatio) + "\n" 
                              "FloorRatio : " + str(EXT_FloorRatio) + "\n"
                              "RatioCentersA : " + str(RatioCentersA) + "\n"
                              "RatioCentersB : " + str(RatioCentersB) + "\n"
                              "RatioCentersC : " + str(RatioCentersB) + "\n"
                              "Count : " + str(CountCorrections) + "/" + str(TotalCorrections) + "\n" 
                              "Percentage : " + str(np.round(CountCorrections/TotalCorrections, 2)*100) + "%"  + "\n"
                              "TimeElapsed : " + str(np.round((Now-TimeStart).seconds/60/60, 2)) + " hours \n")
                                           
                        EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes  = ApplyBDRCorrection(EValuesSource, SubjectDataSet)
                        
                        StdReduction, SubjectDatasetStd, CorrectedDataSetStd = ComputeStdReduction(SourceDataSet, SubjectDataSet, CorrectedDataSet)

                        ScoreArray[a, b, c] = CorrectedDataSetStd

                        if GenerateControledDataSet == True :
                            ControledDataSet, ControledDataSetStd, ErrorStdRatio = ComputeControledDataSet(SubjectDataset, CorrectedDataSet)
                            
                        if GenerateErrorPerAbsolute == True :
                            ErrorPerAbsolute = ComputeErrorPerAbsolute(EAbsolutesSubject, CorrectedDataSet)
                       
                        if PrintMissingEAbsolutes == True :
                            
                            print("MissingEAbsolutes : " + str( MissingEAbsolutes))

                        if TotalDuration == False :
                            TotalDuration = datetime.datetime.now()-Now
                            print("REQUIRED HOURS : " + str(np.round(TotalDuration.seconds*TotalCorrections/60/60, 1)) + "\n" )


            ScoreDict["F"+str(FloorRatio) + " : " + "C" +str(np.round(CeilingRatio, 1))] = ScoreArray
    
    TimeStop = datetime.datetime.now()
    
    OutputName = str(TimeStart)[11:19] + "_to_" + str(TimeStop)[11:19] + " " + Name

    outfile = open(OutputName + "_XYZ.pkl",'wb')
    pickle.dump(ScoreDict,outfile)
    outfile.close()



#%% EVALUATE SCOREDICT & CHECK BEST FOUND SETTINGS

All1 = np.concatenate([DataSet2, DataSet3])
All2 = np.concatenate([DataSet1, DataSet3])
All3 = np.concatenate([DataSet1, DataSet2])

# infile = open("04:12:00_to_06:18:56 DataSet1_VSallothers.pkl",'rb')
# infile = open("22:26:40_to_01:45:33 DataSet3_VSallothers.pkl",'rb')


# infile = open("23:16:31_to_01:39:47 DataSet2_ParametersEvaluation.pkl",'rb')


# infile = open("18:59:39_to_19:08:33 Lsce_VSallothers.pkl",'rb')
# infile = open("19:08:33_to_20:39:09 Airparif_VSallothers.pkl",'rb')
# infile = open("20:39:09_to_02:44:57 Coimbra_VSallothers.pkl",'rb')

# infile = open("18:55:01_to_20:39:18 DataSet1_VSallothers.pkl",'rb')
# infile = open("20:39:18_to_22:26:40 DataSet2_VSallothers.pkl",'rb')



ScoreDict = pickle.load(infile)
infile.close()

DfCorrectedDataSetStd = pd.DataFrame()
DfFloorRatioIdx = pd.DataFrame()

CeilingRatioRange = [0.1, 0.25, 0.5, 1, 2, 3]
FloorRatioRange = [0.05, 0.1, 0.25, 0.5]
RatioCenterRange = [0.1, 0.25, 0.50, 0.75]

for DictKey, ScoreArray in ScoreDict.items() :
    ()
        
    Index = float(DictKey[DictKey.find("F")+1:DictKey.find(" : ")])
    Col = float(DictKey[DictKey.find("C")+1:])

    Min = np.min(ScoreArray)

    a, b, c  = np.where(ScoreArray == Min)

    MaxFloorRatioA = RatioCenterRange[a[0]]
    MaxFloorRatioB = RatioCenterRange[b[0]]
    MaxFloorRatioC = RatioCenterRange[c[0]]
    
    DfCorrectedDataSetStd.loc[Index, Col] = Min
    DfFloorRatioIdx.loc[Index, Col] = str(MaxFloorRatioA) + ", " + str(MaxFloorRatioB) + ", " + str(MaxFloorRatioC)

Min = DfCorrectedDataSetStd.min().min()
print("Min : " + str(Min))
print(DfCorrectedDataSetStd == Min)

print("Please edit SelectedFloor/Ceiling (x/y), depending on matched max values")


##########

# SelectedFloor = 0.5
# SelectedCeiling = 1.0

SelectedFloor = 0.05
SelectedCeiling = 0.5

##########

SelectedMatrix = ScoreDict["F" + str(SelectedFloor) + " : " + "C" + str("3")]

SelectedMatrix[a[0], b[0], c[0]]

ValuesRatioCenter = DfFloorRatioIdx.loc[SelectedFloor, SelectedCeiling].split(",")

##########

Results = {"MinCorrectedStd" : (Min),
"FloorRatio" : (SelectedFloor),
"CeilingRatio" :  (SelectedCeiling),
"RatioCenterA" : (ValuesRatioCenter[0]),
"RatioCenterB" :  (ValuesRatioCenter[1]),
"RatioCenterC" : (ValuesRatioCenter[2])}

print(Results)

#%% CONTROL RESULTS

# SourceDataSet, SubjectDataSet, Name = (All1, DataSet1, "DataSet1")
# SourceDataSet, SubjectDataSet, Name = (All2, DataSet2, "DataSet2")
# SourceDataSet, SubjectDataSet, Name = (All3, DataSet3, "DataSet3")


# SourceDataSet, SubjectDataSet, Name = (DataSet2, DataSet2, "DataSet2")
# SourceDataSet, SubjectDataSet, Name = (DataSet3, DataSet3, "DataSet3")

# SourceDataSet, SubjectDataSet, Name = (Lsce, Lsce, "Lsce")
# SourceDataSet, SubjectDataSet, Name = (Airparif, Airparif, "Airparif")
# SourceDataSet, SubjectDataSet, Name = (Coimbra, Coimbra, "Coimbra")


RatioCentersA = float(ValuesRatioCenter[0])
RatioCentersB =  float(ValuesRatioCenter[1])
RatioCentersC =  float(ValuesRatioCenter[2])

EXT_CeilingRatio =  float(SelectedCeiling)
EXT_FloorRatio =  float(SelectedFloor)

##########
EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
CorrectionDecPre = 0

##########

EXT_RatioCenters = RatioCentersA
EAbsolutes = ComputeCenters(SourceDataSet)

EXT_RatioCenters = RatioCentersB
EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataSet, EAbsolutes)

EXT_RatioCenters = RatioCentersC
EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes  = ApplyBDRCorrection(EValuesSource, SubjectDataSet)

StdReduction, SubjectDatasetStd, CorrectedDataSetStd = ComputeStdReduction(SourceDataSet, SubjectDataSet, CorrectedDataSet)

print(CorrectedDataSetStd)
print(Min)

#%% RUN 10X REDUCTIONS

RatioCentersA = 0.75
RatioCentersB = 0.75
RatioCentersC = 0.75

EXT_CeilingRatio = 0.1
EXT_FloorRatio = 0.05

SplitRatio = 0.4

EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
CorrectionDecPre = 0

# All1 = np.concatenate([DataSet2, DataSet3])
# All2 = np.concatenate([DataSet1, DataSet3])
# All3 = np.concatenate([DataSet1, DataSet2])

SourceDataSet, SubjectDataSet  = All3, DataSet3


ListPercentages = []
ListStd = []

for n in np.arange(0, 10) :
    print("RUN : " + str(n))
    SourceDataSet, SubjectDataSet, DataSet = SplitDataSet(DataSet3)
    
    EXT_RatioCenters = RatioCentersA
    EAbsolutes = ComputeCenters(SourceDataSet)
    
    EXT_RatioCenters = RatioCentersB
    EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataSet, EAbsolutes)
    
    EXT_RatioCenters = RatioCentersC
    EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes  = ApplyBDRCorrection(EValuesSource, SubjectDataSet)
    
    StdReduction, SubjectDatasetStd, CorrectedDataSetStd = ComputeStdReduction(SourceDataSet, SubjectDataSet, CorrectedDataSet)
    
    ListPercentages.append(StdReduction)
    ListStd.append(CorrectedDataSetStd)

np.nanmean(ListPercentages)
np.nanmean(ListStd)

# plt.plot(SubjectDataSet, color = "black", alpha = 0.5)
# plt.plot(CorrectedDataSet, color = "red", alpha = 0.5)

#%% THESIS : COMPARE WITH REFERENCE INSTRUMENT

RatioCentersA = 0.75
RatioCentersB = 0.75
RatioCentersC = 0.75

EXT_CeilingRatio = 0.1
EXT_FloorRatio = 0.05

SplitRatio = 0.4

EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
CorrectionDecPre = 1



EXT_RatioCenters = RatioCentersA

EAbsolutes = Airparif[:,15]
SourceDataSet = Airparif[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]

EXT_RatioCenters = RatioCentersA
# EAbsolutes = ComputeCenters(SourceDataSet)

EXT_RatioCenters = RatioCentersB
EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataSet, EAbsolutes)

EXT_RatioCenters = RatioCentersC
EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes  = ApplyBDRCorrection(EValuesSource, SourceDataSet)

StdReduction, SubjectDatasetStd, CorrectedDataSetStd = ComputeStdReduction(SourceDataSet, SourceDataSet, CorrectedDataSet)



#%% NOISE REDUCTION


start_time = 0
end_time = 1
sample_rate = 1000
time = np.arange(start_time, end_time, 1/sample_rate)
theta = 2
frequency = 2
amplitude = 1


DfNoise = pd.DataFrame()
nSensors = 16
for n in np.arange(1,nSensors):
    noisea = (amplitude + 0.1*n * np.sin(4 * np.pi * frequency * time )) + 20
    
    DfNoise[str(n)] = noisea
    
# sinewave = amplitude * np.sin(2 * np.pi * frequency * time + theta)
# plt.figure(figsize=(20, 6), dpi=80)
plt.plot(DfNoise)



EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
EXT_RatioCenters = 0.5

SubjectDataset = DfNoise
SourceDataset = DfNoise

EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataset)

# EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, OffsetsSum = ApplyBDRCorrection(EValuesSource, DfNoise)
# BenchmarkdAll = DfNoise(DfNoise, DfNoise, CorrectedDataSet)


Values = DfNoise.iloc[10,:]


plt.plot(pd.DataFrame(CorrectedDataSet), color = "blue", lw = 2)
plt.plot(pd.DataFrame(DfNoise), color = "green", lw = 2)

plt.plot(pd.DataFrame(EAbsolutesSource), color = "red", lw = 2)



#%% Print correction profiles

SplitRatio= 0.5


RatioCentersA = 0.75
RatioCentersB = 0.75
RatioCentersC = 0.75

EXT_CeilingRatio = 0.1
EXT_FloorRatio = 0.05

SplitRatio = 0.5

EXT_CenterFunction = "limit"
EXT_InterpMissingEAbsolutes = False
CentersDecPre = 1 # Decimal precision for EValues and EAbsolues, Ex : 0,1,2,3...
CorrectionDecPre = 0

SourceDataSetA, SourceDataSetB, DataSet = SplitDataSet(Airparif[:,:-1])

SourceDataSet = Airparif[:,:-1]


EXT_RatioCenters = RatioCentersA
EAbsolutes = ComputeCenters(SourceDataSet)

EXT_RatioCenters = RatioCentersB
EAbsolutesSource, EValuesSource = ComputeEValuesPerEAbsolutes(SourceDataSet, EAbsolutes)

EXT_RatioCenters = RatioCentersC
EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes  = ApplyBDRCorrection(EValuesSource, SourceDataSet)

StdReduction, SubjectDatasetStd, CorrectedDataSetStd = ComputeStdReduction(SourceDataSet, SourceDataSet, CorrectedDataSet)


# EValuesSourceA = EValuesSource
# EValuesSourceB = EValuesSource


##############

Correction = pd.DataFrame(EValuesSourceA[:, 2:]) 
Names = []
for i in range( 0, len(Correction.T)) :
    Names.append("Cairsens n" + str(i))
Correction.columns = Names

mpl.rcdefaults()
# plt.style.use('seaborn-white')
# print (plt.style.available)

fig = plt.figure()
plt.plot(EValuesSourceB[:, 0], alpha = 1, color = "black", label = "Centre", lw = 2)

for Name in Names:
    print()
    plt.plot(Correction[Name], alpha = 0.75, label = Name)

# plt.legend(bbox_to_anchor=(1, 1.05))


plt.ylabel("Offset capteur")
plt.xlabel("Centre estimé")
# plt.legend()
plt.show()


#%% Plot dispersion reduction


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], lw=4, color = "black"),
                Line2D([0], [0], lw=4, color = "Red"),
                Line2D([0], [0], lw=4, color = "teal")]


plt.plot(SourceDataSet, color = "black", alpha = 1)
plt.plot(CorrectedDataSet[:,1:], color = "Red", alpha = 1)
plt.plot(EAbsolutesSubject, color = "teal", alpha = 1)

