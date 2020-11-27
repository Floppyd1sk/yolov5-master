import pyodbc
import datetime

# CONNECTIONSTRING LOKAL DATABASE

# Definition of connection string used to connect to the database
connString = 'Trusted_Connection=yes;' \
             'Driver={ODBC Driver 17 for SQL Server};' \
             'SERVER=localhost;' \
             'PORT=1433;' \
             'DATABASE=CommentorDB;' \
             'UID=CommentorAdmin;' \
             'PWD=Admin123'


# Gets the latest hour from the database Cars table
def getLatestHour():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT HourId FROM Cars WHERE CarId=(SELECT max(CarId) FROM Cars)')
    for row in cursor:
        # print(row[0])
        return row[0]


# Gets the latest Car amount from the database Cars table
def getLatestCarAmount():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    dateId = getLatestDate()
    dateIdStr = str(dateId)
    hourId = getLatestHour()
    hourIdStr = str(hourId)
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y ")
    hourStr = now.strftime("%H")
    latestRow = getLatestRowCars()
    if hourIdStr == hourStr and dateIdStr == dateStr:
        cursor.execute('SELECT CarAmount FROM Cars WHERE CarId=%s' % latestRow)
        for row in cursor:
            return row[0]
    else:
        return 0

# Gets the latest Car amount from the database Cars table
def getLatestTruckAmount():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    dateId = getLatestDate()
    dateIdStr = str(dateId)
    hourId = getLatestHour()
    hourIdStr = str(hourId)
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y ")
    hourStr = now.strftime("%H")
    latestRow = getLatestRowTrucks()
    if hourIdStr == hourStr and dateIdStr == dateStr:
        cursor.execute('SELECT TruckAmount FROM Trucks WHERE TruckId=%s' % latestRow)
        for row in cursor:
            return row[0]
    else:
        return 0

def getLatestWeek():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT WeekNumber FROM Dates WHERE DateId=(SELECT max(DateId) FROM Dates)')
    for row in cursor:
        return row[0]

def getLatestDate():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT CurrentDate FROM Dates WHERE DateId=(SELECT max(DateId) FROM Dates)')
    for row in cursor:
        return row[0]


# Gets the latest row from the database Cars table
def getLatestRowCars():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT Max(CarId) FROM Cars')
    for row in cursor:
        return row[0]

def getLatestRowTrucks():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT Max(TruckId) FROM Trucks')
    for row in cursor:
        return row[0]


# Gets the latest DateId from Dates table
def getLatestDateId():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    cursor.execute('SELECT Max(DateId) FROM Dates')
    for row in cursor:
        return row[0]


# Checks if the dates table is empty. Returns 1 if it's not empty and 0 if it's empty
def IsDatesEmpty():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    sql = "SELECT count(*) as tot FROM Dates"
    cursor.execute(sql)
    data = cursor.fetchone()
    cn.close()
    if data[0] > 0:
        return 1
    else:
        return 0

# Updates the latest row in Cars table
def updateRow(TableName):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        if TableName == "Cars":
            latestRow = getLatestRowCars()
            newNumber = getLatestCarAmount() + 1
            sql = "Update Cars set CarAmount = %s where CarId = %s" % (newNumber, latestRow)
        elif TableName == "Trucks":
            latestRow = getLatestRowTrucks()
            newNumber = getLatestTruckAmount() + 1
            sql = "Update Trucks set TruckAmount = %s where TruckId = %s" % (newNumber, latestRow)
        cursor = cn.execute(sql)
        cn.commit()
    except Exception as e:
        print('Error connecting to database: ', str(e))

    finally:
        cn.close()
        print("Updated " + TableName + "Id: " + str(latestRow) + " to Amount: " + str(newNumber))

# Inserts a new row into the Cars table
def insertRow(number, check, TableName):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        now = datetime.datetime.now()
        dateStr = now.strftime("%d %b %Y ")
        #dateStr = '06 Oct 2020'
        hourStr = now.strftime("%H")
        print(hourStr)
        #hourStr = '20'                     # Uncomment this to test new day
        year, week_num, day_of_week = now.isocalendar()


        if hourStr == '00' or IsDatesEmpty() == 0 or not dateStr == getLatestDate():  # Checks if Hour is '00', if the Dates table is empty or if it's a new date.
            # If either is true, then it runs the commit
            sql1 = "INSERT INTO Dates(CurrentDate, WeekNumber) VALUES ('%s', '%s' ) " % (dateStr, week_num)
            cursor1 = cn.execute(sql1)
            cn.commit()

        latestDateId = getLatestDateId()
        if TableName == "Cars":
            sql2 = "INSERT INTO Cars(DateId,HourId,CarAmount) VALUES ('%s','%s',%d) " % (latestDateId, hourStr, number)
        elif TableName == "Trucks":
            sql2 = "INSERT INTO Trucks(DateId,HourId,TruckAmount) VALUES ('%s','%s',%d) " % (latestDateId, hourStr, number)
        cursor2 = cn.execute(sql2)


        cn.commit()

    except Exception as e:
        print('Error connecting to database: ', str(e))

    finally:
        cn.close()
        if check:
            print("Table empty, inserted new row")
        else:
            print("Inserted New Row")
