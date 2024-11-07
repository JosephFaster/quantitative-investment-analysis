import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Directorio donde se guardarán los datos descargados
output_dir = r'C:\Users\candy\Downloads\Reto_actinver\Reto actinver 2.0\Data'

# Lista de acciones
# stock_list = [
#     "AA1.MX", "AAL.MX", "AAPL.MX", "AAUN.MX", "ABBV.MX", "ABNB.MX", "AC.MX",
#     "ACTINVRB.MX", "AFRM.MX", "AGNC.MX", "ALFAA.MX", "ALPEKA.MX", "ALSEA.MX",
#     "AMAT.MX", "AMD.MX", "AMXB.MX", "AMZN.MX", "APA.MX", "ASURB.MX", "ATER.MX",
#     "ATOS.MX", "ATVI.MX", "AVGO.MX", "AXP.MX", "BA.MX", "BABAN.MX", "BAC.MX",
#     "BBAJIOO.MX", "BBBY.MX", "BIMBOA.MX", "BMY.MX", "BNGO.MX", "BOLSAA.MX",
#     "BRKB.MX", "BYND.MX", "C.MX", "CAT.MX", "CCL1N.MX", "CEMEXCPO.MX", 
#     "CHDRAUIB.MX", "CLF.MX", "COST.MX", "CPE.MX", "CRM.MX", "CSCO.MX", 
#     "CUERVO.MX", "CVS.MX", "CVX.MX", "DAL.MX", "DIS.MX", "DVN.MX", "ELEKTRA.MX", 
#     "ETSY.MX", "F.MX", "FANG.MX", "FCX.MX", "FDX.MX", "FEMSAUBD.MX", 
#     "FIBRAMQ12.MX", "FIBRAPL14.MX", "FSLR.MX", "FUBO.MX", "FUNO11.MX", 
#     "GAPB.MX", "GCARSOA1.MX", "GCC.MX", "GE.MX", "GENTERA.MX", "GFINBURO.MX", 
#     "GFNORTEO.MX", "GILD.MX", "GM.MX", "GME.MX", "GMEXICOB.MX", "GOLDN.MX", 
#     "GOOGL.MX", "GRUMAB.MX", "HD.MX", "INTC.MX", "JNJ.MX", "JPM.MX", 
#     "KIMBERA.MX", "KO.MX", "KOFUBL.MX", "LABB.MX", "LASITEB-1.MX", "LCID.MX", 
#     "LIVEPOLC-1.MX", "LLY.MX", "LUV.MX", "LVS.MX", "LYFT.MX", "MA.MX", 
#     "MARA.MX", "MCD.MX", "MEGACPO.MX", "MELIN.MX", "META.MX", "MFRISCOA-1.MX", 
#     "MGM.MX", "MRK.MX", "MRNA.MX", "MRO.MX", "MSFT.MX", "MU.MX", "NCLHN.MX", 
#     "NFLX.MX", "NKE.MX", "NKLA.MX", "NUN.MX", "NVAX.MX", "NVDA.MX", "OMAB.MX", 
#     "ORBIA.MX", "ORCL.MX", "OXY1.MX", "PARA.MX", "PBRN.MX", "PE&OLES.MX", 
#     "PEP.MX", "PFE.MX", "PG.MX", "PINFRA.MX", "PINS.MX", "PLTR.MX", "PYPL.MX", 
#     "Q.MX", "QCOM.MX", "RA.MX", "RCL.MX", "RIOT.MX", "RIVN.MX", "ROKU.MX", 
#     "SBUX.MX", "SHOPN.MX", "SITES1A-1.MX", "SKLZ.MX", "SOFI.MX", "SPCE.MX", 
#     "SQ.MX", "T.MX", "TALN.MX", "TERRA13.MX", "TGT.MX", "TELEVISACPO.MX", 
#     "TMO.MX", "TSLA.MX", "TSMN.MX", "TWLO.MX", "TX.MX", "UAL.MX", "UBER.MX", 
#     "UNH.MX", "UPST.MX", "V.MX", "VESTA.MX", "VOLARA.MX", "VZ.MX", "WALMEX.MX", 
#     "WFC.MX", "WISH.MX", "WMT.MX", "WYNN.MX", "X.MX", "XOM.MX", "ZM.MX"
# ]
stock_list =['']

# Especificar el rango de fechas
start_date = "2022-01-01"  # Cambia la fecha de inicio aquí
end_date = "2024-11-06"    # Cambia la fecha de fin aquí

# Crear directorio si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para descargar los datos
def download_stock_data(stock):
    try:
        data = yf.download(stock, start=start_date, end=end_date)
        if data.empty:
            print(f"No se encontraron datos para {stock}. Revisa si el ticker es correcto.")
            return None
        return data
    except Exception as e:
        print(f"Error descargando datos para {stock}: {e}")
        return None

# Descargar datos y guardar en archivos CSV
for stock in stock_list:
    print(f"Descargando datos para {stock}...")
    data = download_stock_data(stock)
    if data is not None:
        file_path = os.path.join(output_dir, f"{stock}.csv")
        data.to_csv(file_path)  # Guardar datos como CSV
        print(f"Datos guardados en {file_path}")
    else:
        print(f"No se guardaron datos para {stock} debido a un error o ticker incorrecto.")
