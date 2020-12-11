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
    sql = 'SELECT HourStamp FROM Vehicle WHERE VehicleId = (SELECT max(VehicleId) FROM Vehicle)'
    cursor.execute(sql)
    for row in cursor:
        # print(row[0])
        return row[0]

def getLatestDate():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    sql = 'SELECT DateStamp FROM Vehicle WHERE VehicleId = (SELECT max(VehicleId) FROM Vehicle)'
    cursor.execute(sql)
    for row in cursor:
        return row[0]


# Gets the latest Car amount from the database Cars table
def getLatestVehicleAmount(vehicleType):
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    dateId = getLatestDate()
    dateIdStr = str(dateId)
    hourId = getLatestHour()
    hourIdStr = str(hourId)
    now = datetime.datetime.now()
    dateStr = now.strftime("%d %b %Y")
    hourStr = now.strftime("%H")
    if hourIdStr == hourStr and dateIdStr == dateStr and CheckIfTypeExists(vehicleType, hourStr, dateStr) == 1:
        sql = "SELECT VehicleAmount FROM Vehicle WHERE TypeName = '%s' and HourStamp = %s" % (vehicleType, hourStr)
        cursor.execute(sql)
        for row in cursor:
            return row[0]
    else:
        return 0

def getLatestWeek():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    latestRow = getLatestRow()
    sql = 'SELECT WeekNumber FROM Vehicle WHERE VehicleId = %s' % latestRow
    cursor.execute(sql)
    for row in cursor:
        return row[0]

# Gets the latest row from the database Cars table
def getLatestRow():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    sql = 'SELECT Max(VehicleId) FROM Vehicle'
    cursor.execute(sql)
    for row in cursor:
        return row[0]

# Checks if the dates table is empty. Returns 1 if it's not empty and 0 if it's empty
def IsDatesEmpty():
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    sql = "SELECT count(*) as total FROM Dates"
    cursor.execute(sql)
    data = cursor.fetchone()
    cn.close()
    if data[0] > 0:
        return 1
    else:
        return 0

def IsVehiclesEmpty(vehicleType):
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    sql = "SELECT count(*) as total FROM Vehicle where TypeName = '%s'" % vehicleType
    cursor.execute(sql)
    data = cursor.fetchone()
    cn.close()
    if data[0] > 0:
        return 1
    else:
        return 0

def CheckIfTypeExists(vehicleType, hourStamp, dateStamp):
    cn = pyodbc.connect(connString, autocommit=True)
    cursor = cn.cursor()
    #sql = "SELECT TOP 1 VehicleId, TypeName, HourStamp from Vehicle where TypeName = '%s' and HourStamp = %s ORDER BY VehicleId DESC" % (typeName, hourStamp)
    sql = "SELECT count(*) as total from Vehicle where TypeName = '%s' and HourStamp = %s and DateStamp = '%s'"  % (vehicleType, hourStamp, dateStamp)
    cursor.execute(sql)
    data = cursor.fetchone()
    cn.close()
    if data[0] > 0:
        return 1
    else:
        return 0

# Updates the latest row in Cars table
def updateRow(vehicleType):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        latestRow = getLatestRow()
        newNumber = getLatestVehicleAmount(vehicleType) + 1
        now = datetime.datetime.now()
        dateStr = now.strftime("%d %b %Y")
        hourStr = now.strftime("%H")
        sql = "Update Vehicle set VehicleAmount = %s WHERE TypeName = '%s' and HourStamp = %s and DateStamp = '%s'" % (newNumber, vehicleType, hourStr, dateStr)

        cursor = cn.execute(sql)
        cn.commit()
    except Exception as e:
        print('Error connecting to database: ', str(e))

    finally:
        cn.close()
        print("Updated VehicleId ("+str(vehicleType)+"): " + str(latestRow) + " to Amount: " + str(getLatestVehicleAmount(vehicleType)))

# Inserts a new row into the Cars table
def insertRow(number, check, vehicleType):
    try:
        cn = pyodbc.connect(connString, autocommit=True)
        now = datetime.datetime.now()
        dateStr = now.strftime("%d %b %Y")
        #dateStr = '06 Oct 2020'
        hourStr = now.strftime("%H")
        #hourStr = '20'                     # Uncomment this to test new day
        year, week_num, day_of_week = now.isocalendar()
        feed = "1"
        sql2 = "INSERT INTO Vehicle(TypeName, VehicleAmount, Feed, DateStamp, WeekNumber, HourStamp) VALUES ('%s','%d',%s,'%s','%d','%s') " % (vehicleType, number, feed, dateStr, week_num, hourStr)
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
