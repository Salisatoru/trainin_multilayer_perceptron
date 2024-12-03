from fuzzy_logic import *


def main():
    occupancyFactor = float(input('Enter occupancy factor: ')) * 100
    averageDistance = float(input('Enter average distance: ')) * 100
    trafficIntensity = float(input('Enter traffic intensity factor: ')) * 100
    rules = evaluateRules(occupancyFactor, averageDistance, trafficIntensity)
    outputMfs = {'vs': getVSPlots(), 's': getSPlots(), 'rs': getRSPlots(), 'm': getMPlots(),
                 'rl': getRLPlots(), 'l': getLPlots(), 'vl': getVLPlots()
                 }
    aggregatedPlots = fisAggregation(rules, outputMfs)
    centroid = getCentroid(aggregatedPlots) / 100
    print(centroid)


def fisAggregation(rules, outputMfs):
    vs = outputMfs['vs']
    s = outputMfs['s']
    rs = outputMfs['rs']
    m = outputMfs['m']
    rl = outputMfs['rl']
    l = outputMfs['l']
    vl = outputMfs['vl']
    aggregatePlots = [0] * 100
    for rule in range(len(rules)):
        for i in range(100):
            if rules[rule][0] > 0 and i < 20:
                aggregatePlots[i] = min(rules[rule][0], vs[i])
            if rules[rule][1] > 0 and i > 15 and i < 35:
                aggregatePlots[i] = min(rules[rule][1], s[i])
            if rules[rule][2] > 0 and i > 30 and i < 45:
                aggregatePlots[i] = min(rules[rule][2], rs[i])
            if rules[rule][3] > 0 and i > 40 and i < 60:
                aggregatePlots[i] = min(rules[rule][3], m[i])
            if rules[rule][4] > 0 and i > 55 and i < 70:
                aggregatePlots[i] = min(rules[rule][4], rl[i])
            if rules[rule][5] > 0 and i > 65 and i < 85:
                aggregatePlots[i] = min(rules[rule][5], l[i])
            if rules[rule][6] > 0 and i > 80:
                aggregatePlots[i] = min(rules[rule][6], vl[i])
    return aggregatePlots


def evaluateRules(occupancyFactor, averageDistance, trafficIntensity):
    """
        rowSize = 27 ; rules
        colSize = 7  ; membership functions of output variable "n"
    """
    rules = [[0] * 7 for i in range(27)]
    """
        Definitions
        Input "m": occupancy factor
            ml - low
            mm - medium
            mh - high
        
        Input "s": average distance
            ss - short
            sm - medium
            sl - long
        Input "p": traffic intensity
            pl - low
            pm - medium
            ph - high
    """
    ml = fuzzifyOccupancyLow(occupancyFactor)
    mm = fuzzifyOccupancyMedium(occupancyFactor)
    mh = fuzzifyOccupancyHigh(occupancyFactor)

    ss = fuzzifyAverageDistanceShort(averageDistance)
    sm = fuzzifyAverageDistanceMedium(averageDistance)
    sl = fuzzifyAverageDistanceLong(averageDistance)

    pl = fuzzifyTrafficIntensityLow(trafficIntensity)
    pm = fuzzifyTrafficIntensityMedium(trafficIntensity)
    ph = fuzzifyTrafficIntensityHigh(trafficIntensity)

    """
        MembershipOutputIndex:
            VS - 0
            S - 1
            RS - 2
            ...
            VL - 6
        For all "n" with output VS, store it in column 0, and for S in column 1 ...
    """
    # rules[ruleIndex][membershipOutputIndex]
    rules[0][0] = min(min(ml, ss), pl)
    rules[1][0] = min(min(mm, ss), pl)
    rules[2][0] = min(min(mh, ss), pl)
    rules[3][0] = min(min(ml, sm), pl)
    rules[4][0] = min(min(mm, sm), pl)
    rules[5][0] = min(min(mh, sm), pl)
    rules[6][1] = min(min(ml, sl), pl)
    rules[7][1] = min(min(mm, sl), pl)
    rules[8][0] = min(min(mh, sl), pl)
    rules[9][1] = min(min(ml, ss), pm)
    rules[10][0] = min(min(mm, ss), pm)
    rules[11][0] = min(min(mh, ss), pm)
    rules[12][2] = min(min(ml, sm), pm)
    rules[13][1] = min(min(mm, sm), pm)
    rules[14][0] = min(min(mh, sm), pm)
    rules[15][1] = min(min(ml, sl), pm)
    rules[16][2] = min(min(mm, sl), pm)
    rules[17][1] = min(min(mh, sl), pm)
    rules[18][6] = min(min(ml, ss), ph)
    rules[19][5] = min(min(mm, ss), ph)
    rules[20][3] = min(min(mh, ss), ph)
    rules[21][3] = min(min(ml, sm), ph)
    rules[22][3] = min(min(mm, sm), ph)
    rules[23][1] = min(min(mh, sm), ph)
    rules[24][4] = min(min(ml, sl), ph)
    rules[25][3] = min(min(mm, sl), ph)
    rules[26][2] = min(min(mh, sl), ph)
    return rules


def fuzzifyOccupancyLow(occupancyFactor):
    return trapmf(occupancyFactor, [0, 0, 20, 40])


def fuzzifyOccupancyMedium(occupancyFactor):
    return trimf(occupancyFactor,  [20, 50, 80])


def fuzzifyOccupancyHigh(occupancyFactor):
    return trapmf(occupancyFactor, [60, 80, 100, 100])


def fuzzifyAverageDistanceShort(averageDistance):
    return trapmf(averageDistance, [0, 0, 20, 40])


def fuzzifyAverageDistanceMedium(averageDistance):
    return trimf(averageDistance, [20, 50, 80])


def fuzzifyAverageDistanceLong(averageDistance):
    return trapmf(averageDistance, [60, 80, 100, 100])


def fuzzifyTrafficIntensityLow(trafficIntensity):
    return trapmf(trafficIntensity, [0, 0, 20, 40])


def fuzzifyTrafficIntensityMedium(trafficIntensity):
    return trimf(trafficIntensity, [20, 50, 80])


def fuzzifyTrafficIntensityHigh(trafficIntensity):
    return trapmf(trafficIntensity, [60, 80, 100, 100])


def getVSPlots():
    return getTrapmfPlots(0, 100, [0, 0, 10, 20], "left")


def getSPlots():
    return getTrimfPlots(0, 100, [15, 25, 35])


def getRSPlots():
    return getTrimfPlots(0, 100, [30, 35, 45])


def getMPlots():
    return getTrimfPlots(0, 100, [40, 50, 60])


def getRLPlots():
    return getTrimfPlots(0, 100, [55, 65, 70])


def getLPlots():
    return getTrimfPlots(0, 100, [65, 75, 85])


def getVLPlots():
    return getTrapmfPlots(0, 100, [80, 90, 100, 100], "right")


if __name__ == '__main__':
    main()