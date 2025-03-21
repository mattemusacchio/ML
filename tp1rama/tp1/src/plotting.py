import matplotlib.pyplot as plt
import seaborn as sns

def plot_lat_lon(dev_df):
    """Genera un gráfico de dispersión de latitud y longitud coloreado por unidad de medida y devuelve coordenadas promedio."""
    fig, ax = plt.subplots(figsize=(4, 4))
    scatter = ax.scatter(dev_df['lon'], dev_df['lat'], c=dev_df['area_units'].astype('category').cat.codes)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Lat. vs Lon. by Unit')
    handles, labels = scatter.legend_elements()
    ax.legend(handles, dev_df['area_units'].unique(), title='Area Units')
    plt.show()
    return dev_df.groupby('area_units')[['lat', 'lon']].agg(['mean', 'std'])