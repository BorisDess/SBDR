
def ComputeCenters(Data):

    DataCenters = np.full(Data.shape[0], np.nan)
    Data = np.array(Data)

    n = 0
    if EXT_CenterFunction == "limit":
        for n in np.arange(0, Data.shape[0]):
            Values = Data[n, :]
            DataCenters[n] = GetCenterLimit(Values)
            
    elif EXT_CenterFunction == "legacy":
        for n in np.arange(0, Data.shape[0]):
            Values = Data[n, :]
            DataCenters[n] = GetCenterLegacy(Values)
            
    elif EXT_CenterFunction == "median":
        for n in np.arange(0, Data.shape[0]):
            Values = Data[n, :]
            DataCenters[n] = GetCenterMedian(Values)

    else :
        print("No function")
    return(DataCenters)

def GetCenterLimit(Values):
    
    global EXT_RatioCenters
     
    Len = len(Values)
    if Len == 0: 
        Center = np.nan
    
    elif Len  == 1: 
        Center = np.nanmean(Values[0])
    elif Len == 2: 
        Center =  np.nanmean(Values)
   
    else :
        ListDeltas = []
        for idx in np.arange(0, len(Values)):
            Deltas = np.full(len(Values), np.nan)
            Value = Values[idx]
            if np.isnan(Value) :
                ListDeltas.append(Deltas)
            else:
                Selection = np.full(len(Values), False)
                Selection[idx] = True
                SelectedValues = Values[~Selection]
                Deltas[~Selection] = np.abs(SelectedValues - Value)
                ListDeltas.append(Deltas)
                
        Sum = np.nansum(ListDeltas, axis = 1)
        Max = np.nanmax(ListDeltas, axis = 1)
        Result = np.divide(Sum,Max)
    
        Centers = Values[Result.argsort()]
        Centers = Centers[~np.isnan(Centers)]
        SelectedSensors = int(np.round(EXT_RatioCenters*len(Centers)))
        if SelectedSensors == 0 :
            SelectedSensors = 1
                
        SelectedCenters = Centers[0:SelectedSensors]
        
        Center = np.mean(SelectedCenters)
        Center =  np.round(np.mean(Center), CentersDecPre)

    return(Center)

def GetCenterMedian(Values):
    
    if len(Values) == 0: 
        Center = np.nan
    
    elif len(Values) == 1: 
        Center =  np.round(np.nanmean(Values[0]), CentersDecPre)
    elif len(Values) == 2: 
        Center =  np.round(np.nanmean(Values), CentersDecPre)
        
    elif all(x == Values[0] for x in Values): 
        Center =  np.round(np.nanmean(Values[0]), CentersDecPre)
    
    elif sum(np.isfinite(Values)) == 0 : 
        Center = np.nan
    else :
        
        Center = np.nanmedian(Values)
        Center =  np.round(np.nanmean(Values), CentersDecPre)

    return(Center)

def GetCenterLegacy(Values):    
    if len(Values) == 0: 
        Center = np.nan
    
    elif len(Values) == 1:
        Center = np.round(Values[0], CentersDecPre)
        
    elif len(Values) == 2: 
        Center =  np.round(np.nanmean(Values), CentersDecPre)
        
    elif all(x == Values[0] for x in Values): 
        Center = np.round(Values[0], CentersDecPre)
    
    elif sum(np.isfinite(Values)) == 0 :
        Center = np.nan
        
    else :
        
        Values = np.array(Values)
        zScore = np.divide(Values - np.nanmedian(Values), np.nanstd(Values))
        
        zScoreAbs = np.abs(zScore)

        if all(x == zScoreAbs[0] for x in zScoreAbs): # If all values diverge the same
            Center = np.round(Values.mean(), CentersDecPre)
            return(np.round(Center, CentersDecPre))
            
        MaxzScore = np.nanmax(zScoreAbs)
        CenteringBound = MaxzScore-(MaxzScore*EXT_NormBound) # TO DEFINE
        zScoreSquareNormalized = (VectorizedSigmoid((zScoreAbs/CenteringBound)+0.5)-0.5)
        Selection = zScoreSquareNormalized<=EXT_Threshold
        SelectedValues = Values[np.array(Selection)]

        if all(x == zScoreSquareNormalized[0] for x in zScoreSquareNormalized):
            Center = np.median(Values)

        elif sum(Selection) == 0:
            Center = np.nanmedian(Values)
        
        elif all(x == SelectedValues[0] for x in SelectedValues):
            Center = SelectedValues[0]

        else :
            ZScoreWeight = np.true_divide(SelectedValues, SelectedValues.sum())
            Center = sum(SelectedValues*ZScoreWeight)

        Center = np.round(Center, CentersDecPre)
    return(Center)


def ComputeEValuesPerEAbsolutes(DataSet, EAbsolutes):

    EAbsolutesUnique = pd.Series(np.unique(EAbsolutes)).dropna().values

    EValues = np.full([EAbsolutesUnique.shape[0], DataSet.shape[1] + 2], np.nan) 
    EValues[:, 0] = EAbsolutesUnique
    
    n = 0
    for EAbsolute in EAbsolutesUnique :

        BooleanSelection = EAbsolutes == EAbsolute 
        EValues[n, 1] = BooleanSelection.sum()

        Selected = DataSet[BooleanSelection]
        EValues[n, 2:] = ComputeCenters(Selected.T)
        
        n = n+1
        
    EValues[:,0] = np.round(EValues[:,0], CorrectionDecPre)
    EAbsolutesUnique = np.unique(EValues[:,0])
    EValuesRounded = np.full([EAbsolutesUnique.shape[0], DataSet.shape[1] + 2], np.nan) 
    EValuesRounded[:, 0] = EAbsolutesUnique

    for EAbsolute in EAbsolutesUnique:
        SelectedEAbsolute = EValues[EValues[:,0] == EAbsolute]
        MultipliedCorrection = (SelectedEAbsolute[:,2:].T*SelectedEAbsolute[:,1]).T
        MultipliedCorrectionSum = MultipliedCorrection.sum(axis = 0)
        Count = SelectedEAbsolute[:,1].sum()
        EValuesRounded[ EAbsolutesUnique == EAbsolute , 2:] = np.divide(MultipliedCorrectionSum,Count).T
        EValuesRounded[ EAbsolutesUnique == EAbsolute , 1] = Count

        
        EValuesRounded[:, 1:] = np.round(EValuesRounded[:, 1:], CentersDecPre)
    

    return(EAbsolutes, EValuesRounded)


def ApplyBDRCorrection(EValuesSource, SubjectDataSet):

    Ceiling = EXT_CeilingRatio*np.median(EValuesSource[:,1])
    Floor = (Ceiling*EXT_FloorRatio)
    RampSize =  Ceiling-Floor
    
    EAbsolutesSubject = ComputeCenters(SubjectDataSet)
    
    if CorrectionDecPre != CentersDecPre :
        EAbsolutesSubject = np.round(EAbsolutesSubject, CorrectionDecPre)
    
    CorrectedDataSet = np.full([SubjectDataSet.shape[0], SubjectDataSet.shape[1]+1], np.nan)
    CorrectedDataSet[:, 0] = EAbsolutesSubject   

    EValueSubjectMax = EAbsolutesSubject[~np.isnan(EAbsolutesSubject)].max()
    EValueMaxOccurence = EValuesSource[:, 1].max()

    MissingEAbsolutes = []
    
    if EXT_InterpMissingEAbsolutes :
        EValuesSourceInterpolated = pd.DataFrame(index = np.arange(0, EValueSubjectMax), columns = np.arange(0, EValuesSource.shape[1]-1), data = np.nan)
        for Data in EValuesSource :
            ()
            EValuesSourceInterpolated.loc[Data[0], :] = Data[1:]

        EValuesSourceInterpolated = EValuesSourceInterpolated.loc[EValuesSourceInterpolated.dropna().index.min():EValuesSourceInterpolated.dropna().index.max(),:]
        EValuesSourceInterpolated = EValuesSourceInterpolated.interpolate(method='linear', limit_direction='both', axis=0).round(CentersDecPre)
        
        EValuesSource = EValuesSourceInterpolated.reset_index().to_numpy()

    else:
        for EAbsolute in np.unique(EAbsolutesSubject):
            if EAbsolute not in EValuesSource[:,0]:
                MissingEAbsolutes.append(EAbsolute)
                  

    NoCorrectionApplied = []
    UniqueEAbsolutesSubject = np.unique(EAbsolutesSubject)
    for EAbsolute in UniqueEAbsolutesSubject :
        
        SelectedEValuesSource = EValuesSource[:, 0] == EAbsolute

        SelectedDatasetSubject = EAbsolutesSubject == EAbsolute

        if (SelectedEValuesSource).sum() == 1 :
            
            EValueCount = EValuesSource[:, 1][SelectedEValuesSource][0]            
            
            OffSet = EValuesSource[SelectedEValuesSource][0][2:]-EAbsolute
            
            if (EValueCount < Ceiling) :
                RatioOffset = 1 - np.divide((EValueCount - Floor), RampSize)
                
                if RatioOffset > 1 :
                    
                    RatioOffset = 1
                    AdjustedOffSet = OffSet - (OffSet*RatioOffset)

                else :
                    
                    AdjustedOffSet = OffSet - (OffSet*RatioOffset)
                
            else :
                AdjustedOffSet = OffSet
            
            CorrectedDataSet[SelectedDatasetSubject, 1:] = SubjectDataSet[SelectedDatasetSubject] - AdjustedOffSet
        
        else :

            NoCorrectionApplied.append(EAbsolute)
            CorrectedDataSet[SelectedDatasetSubject, 1:] = SubjectDataSet[SelectedDatasetSubject]
            
    return(EAbsolutesSubject, CorrectedDataSet, NoCorrectionApplied, MissingEAbsolutes)

def ComputeControledDataSet(SubjectDataset, CorrectedDataSet):

    
    if ComputeControledDataSet == True : 
        ControledDataSet = np.full(CorrectedDataSet.shape + 1, np.nan)
        ControledDataSet[:,0] = CorrectedDataSet[:,0]
        
        for n in np.arange(0, SubjectDataset.shape[0]):
            ()      
            if np.nanstd(SubjectDataset[n,:]) <= np.nanstd(CorrectedDataSet[n,1:]): 
                ControledDataSet[n,2:] = SubjectDataset[n,:]
                ControledDataSet[n,1:] = "BadCorrection"
                
            else :
                ControledDataSet[n,2:] = CorrectedDataSet[n,1:]
                ControledDataSet[n,1:] = "GoodCorrection"

            ControledDataSetStd = np.nanmean(np.nanstd(ControledDataSet[:, 1:], axis = 1))
            ErrorStdRatio = 100/(CorrectedDataSetStd / (CorrectedDataSetStd - ControledDataSetStd))

    return(ControledDataSet, ControledDataSetStd, ErrorStdRatio)

def ComputeErrorPerAbsolute(EAbsolutesSubject, CorrectedDataSet):
    UniqueEAbsolutesSubject = np.unique(EAbsolutesSubject)
    ErrorPerAbsolute = np.full((len(UniqueEAbsolutesSubject), CorrectedDataSet.shape[1]+1), np.nan)
    ErrorPerAbsolute[:,0] = UniqueEAbsolutesSubject
    for n in np.arange(0, len(UniqueEAbsolutesSubject)):
        Unique = UniqueEAbsolutesSubject[n]
        UniqueData = CorrectedDataSet[CorrectedDataSet[:,0] == Unique][:,1:]
        ErrorPerAbsolute[n,1:] = UniqueData.shape[0]
        ErrorPerAbsolute[n,2:] = np.nansum(np.abs(UniqueData - Unique),0)
    
    ErrorPerAbsolute[:,2:] = np.divide(ErrorPerAbsolute[:,2:].T, ErrorPerAbsolute[:,1]).T
        
    return(ErrorPerAbsolute)

def ComputeStdReduction(SourceDataSet, SubjectDataset, CorrectedDataSet):
    
    SubjectDatasetStd = np.nanmean(np.std(SubjectDataset[:, :], axis = 1))

    CorrectedDataSetStd = np.nanmean(np.nanstd(CorrectedDataSet[:, 1:], axis = 1))

    ReductionSign = np.sign((SubjectDatasetStd - CorrectedDataSetStd))
    StdReduction = ReductionSign*(100/ (SubjectDatasetStd / np.abs((SubjectDatasetStd - CorrectedDataSetStd))  ) )
    
    return(StdReduction, SubjectDatasetStd, CorrectedDataSetStd)

def NormalisedSigmoid(x):
    y = ((2*(x-0.5)-2*(x-0.5)*EXT_Sigmoid) / (2*(EXT_Sigmoid-(np.abs(2*(x-0.5)))*2*EXT_Sigmoid+1))) + 0.5
    return(y)

def VectorizedSigmoid(Array):
    
    Output = []
    for Element in Array:
        if Element > 1 :
            Element = 1
        if Element < 0 :
            Element = 0
            
        Output.append(NormalisedSigmoid(Element))
    
    return(np.array(Output))

