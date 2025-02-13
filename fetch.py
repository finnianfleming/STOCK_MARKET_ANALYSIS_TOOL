import yfinance as yf  # Library for fetching stock data
import pandas as pd  # Library for handling data in tables (like Excel)
import dash  # Library for creating a web-based dashboard
from dash import dcc, html  # Components for the Dash dashboard
import plotly.graph_objects as go  # For creating interactive graphs
import torch  # PyTorch for building and training the prediction model
import torch.nn as nn  # For creating neural network layers
from sklearn.preprocessing import MinMaxScaler  # For scaling data (e.g., making numbers smaller)
from rich.console import Console  # For printing colorful and formatted outputs in the terminal
from rich.table import Table  # For displaying data in a table format in the terminal
import numpy as np  # For working with numerical data and arrays

# Setting up global variables for later use
console = Console()  # A console object to print colorful messages
scaler = MinMaxScaler()  # A scaler object to normalize data (e.g., make values between 0 and 1)

# Defining a neural network for predicting stock prices
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Set up the layers of the neural network."""
        super(StockPredictor, self).__init__()
        self.hidden_layer = nn.Linear(input_size, hidden_size)  # Hidden layer
        self.output_layer = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        """Defines how the input data flows through the network."""
        x = torch.relu(self.hidden_layer(x))  # Apply activation function to hidden layer
        x = self.output_layer(x)  # Compute the final output
        return x


def fetch_stock_data(ticker, start_date=None, end_date=None):
    """
    Fetch stock data using yfinance.
    Ticker is the stock symbol (e.g., AAPL for Apple).
    """
    console.print(f"\n[bold green]Fetching stock data for {ticker}...[/bold green]")
    try:
        stock = yf.Ticker(ticker)  # Initialize the ticker
        # Get historical data for the stock
        if start_date and end_date:
            data = stock.history(start=start_date, end=end_date)
        else:
            data = stock.history(period="1y")  # Default: Last 1 year
        if data.empty:
            # If no data, show a message
            console.print(f"[bold red]No data found for ticker '{ticker}'. Please check the ticker or date range.[/bold red]")
        return data
    except Exception as e:
        # Handle any errors while fetching the data
        console.print(f"[bold red]Error fetching data: {e}[/bold red]")
        return pd.DataFrame()


def calculate_indicators(data):
    """
    Calculate indicators like EMA, Bollinger Bands, and RSI.
    These are tools that help analyze stock prices.
    """
    console.print("\n[bold blue]Calculating technical indicators...[/bold blue]")
    try:
        # Exponential Moving Averages (EMA)
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()  # 20-day average
        rolling_std = data['Close'].rolling(window=20).std()  # 20-day standard deviation
        data['Upper_Band'] = rolling_mean + (rolling_std * 2)  # Upper band
        data['Lower_Band'] = rolling_mean - (rolling_std * 2)  # Lower band

        # Relative Strength Index (RSI)
        delta = data['Close'].diff()  # Change in price
        gain = delta.where(delta > 0, 0)  # Positive change
        loss = -delta.where(delta < 0, 0)  # Negative change
        avg_gain = gain.rolling(window=14).mean()  # Average gain over 14 days
        avg_loss = loss.rolling(window=14).mean()  # Average loss over 14 days
        rs = avg_gain / avg_loss  # Relative strength
        data['RSI'] = 100 - (100 / (1 + rs))  # RSI formula

        console.print("[bold green]Indicators successfully calculated![/bold green]\n")
        return data
    except Exception as e:
        # Handle errors while calculating indicators
        console.print(f"[bold red]Error calculating indicators: {e}[/bold red]")
        return data


def provide_insights(data):
    """
    Print a summary of the stock data in the terminal.
    Includes metrics like average price and RSI.
    """
    console.print("\n[bold magenta]--- Stock Insights ---[/bold magenta]")
    try:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")  # Metric name
        table.add_column("Value")  # Metric value

        # Add rows with calculated values
        table.add_row("Average Closing Price", f"${data['Close'].mean():.2f}")
        table.add_row("Highest Closing Price", f"${data['Close'].max():.2f}")
        table.add_row("Lowest Closing Price", f"${data['Close'].min():.2f}")
        table.add_row("Volatility (Std Dev)", f"{data['Close'].std():.2f}")

        # Add RSI insights
        if 'RSI' in data.columns:
            last_rsi = data['RSI'].iloc[-1]
            if last_rsi > 70:
                table.add_row("RSI", f"{last_rsi:.2f} (Overbought)")  # RSI > 70 means overbought
            elif last_rsi < 30:
                table.add_row("RSI", f"{last_rsi:.2f} (Oversold)")  # RSI < 30 means oversold
            else:
                table.add_row("RSI", f"{last_rsi:.2f} (Neutral)")
        else:
            table.add_row("RSI", "Not enough data")

        # Print the table
        console.print(table)
    except Exception as e:
        console.print(f"[bold red]Error providing insights: {e}[/bold red]")


def export_data(data, ticker):
    """
    Export stock data to a file (CSV, Excel, or JSON).
    """
    export_format = input("Choose an export format (CSV, Excel, or JSON): ").strip().lower()
    try:
        # Ensure the data has no timezone info
        if pd.api.types.is_datetime64_any_dtype(data.index):
            data.index = data.index.tz_localize(None)

        # Export based on user choice
        if export_format == "csv":
            data.to_csv(f"{ticker}_analysis.csv")
            console.print(f"[bold green]Data successfully saved to {ticker}_analysis.csv.[/bold green]")
        elif export_format == "excel":
            try:
                data.to_excel(f"{ticker}_analysis.xlsx")
                console.print(f"[bold green]Data successfully saved to {ticker}_analysis.xlsx.[/bold green]")
            except ModuleNotFoundError:
                console.print("[bold red]Error: 'openpyxl' library is required for Excel export.[/bold red]")
        elif export_format == "json":
            data.to_json(f"{ticker}_analysis.json")
            console.print(f"[bold green]Data successfully saved to {ticker}_analysis.json.[/bold green]")
        else:
            console.print("[bold red]Invalid format. Export skipped.[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error exporting data: {e}[/bold red]")


def predict_stock_prices(data):
    """
    Use a PyTorch neural network to predict future stock prices.
    Includes additional features and improved model architecture.
    """
    console.print("\n[bold blue]--- Predicting Stock Prices ---[/bold blue]")
    try:
        # Add additional features for training
        data['Days'] = (data.index - data.index[0]).days
        data['Pct_Change'] = data['Close'].pct_change()
        data['Volume_Change'] = data['Volume'].pct_change()
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data = data.dropna()  # Drop rows with NaN after feature calculation

        # Define features and target
        features = ['Days', 'Close', 'EMA_20', 'EMA_50', 'RSI', 'Upper_Band', 'Lower_Band', 'Pct_Change', 'Volume_Change', 'Momentum']
        scaled_features = scaler.fit_transform(data[features].values)
        X = scaled_features[:, :-1]  # Features (exclude Close)
        y = scaled_features[:, -1]  # Target (scaled Close price)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Add dimension

        # Train/Test split
        train_size = int(len(X_tensor) * 0.8)
        X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

        # Define the improved model
        model = StockPredictor(input_size=X_train.shape[1], hidden_size=128, output_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        console.print("[cyan]Training the neural network...[/cyan]")
        for epoch in range(500):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()

            # Print loss every 50 epochs
            if (epoch + 1) % 50 == 0:
                console.print(f"Epoch {epoch + 1}/500 - Loss: {loss.item():.4f}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_test)
            val_loss = torch.sqrt(criterion(val_predictions, y_test))
            console.print(f"Validation RMSE: {val_loss.item():.4f}")

        # Predict future prices
        future_days = np.array(range(data['Days'].max() + 1, data['Days'].max() + 31)).reshape(-1, 1)
        future_features = np.hstack([
            future_days,
            np.tile(data[['EMA_20', 'EMA_50', 'RSI', 'Upper_Band', 'Lower_Band', 'Pct_Change', 'Volume_Change', 'Momentum']].iloc[-1].values, (30, 1))
        ])
        future_scaled = scaler.transform(np.hstack([future_features, np.zeros((30, 1))]))
        future_tensor = torch.tensor(future_scaled[:, :-1], dtype=torch.float32)

        # Generate predictions
        future_prices_scaled = model(future_tensor).detach().numpy()
        future_prices = scaler.inverse_transform(np.hstack([future_features, future_prices_scaled]))[:, -1]

        # Generate future dates
        future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='D')

        # Combine predictions into a DataFrame
        predictions = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices.flatten()})
        return predictions
    except Exception as e:
        console.print(f"[bold red]Error in prediction: {e}[/bold red]")
        return pd.DataFrame()





def create_dashboard(data, ticker, predictions):
    """
    Build and launch a web dashboard for stock analysis using Dash.
    """
    app = dash.Dash(__name__)

    # Add buy/sell signals based on EMA
    data['Buy_Signal'] = (data['Close'] > data['EMA_20']) & (data['Close'].shift(1) <= data['EMA_20'])
    data['Sell_Signal'] = (data['Close'] < data['EMA_20']) & (data['Close'].shift(1) >= data['EMA_20'])

    # Create a graph with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close Price"))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_20'], name="EMA 20", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name="Upper Bollinger Band", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name="Lower Bollinger Band", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(
        x=data[data['Buy_Signal']].index,
        y=data[data['Buy_Signal']]['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(symbol='triangle-up', color='green', size=10)
    ))
    fig.add_trace(go.Scatter(
        x=data[data['Sell_Signal']].index,
        y=data[data['Sell_Signal']]['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(symbol='triangle-down', color='red', size=10)
    ))
    fig.update_layout(
        title=f"{ticker} Stock Analysis with Indicators and Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Indicators"
    )

    # Add predictions to the dashboard
    app.layout = html.Div([
        html.H1(f"{ticker} Stock Dashboard"),
        html.Div("Visualizing stock data and predictions."),
        dcc.Graph(figure=fig),
        html.H2("Predicted Prices for the Next 30 Days"),
        dcc.Graph(figure=go.Figure(data=[
            go.Scatter(x=predictions['Date'], y=predictions['Predicted Price'], name="Predicted Price")
        ]))
    ])

    # Start the Dash app
    app.run_server(debug=True, use_reloader=False)


def main():
    """
    Main function to run the stock analysis tool.
    """
    # Display a styled project name at the start
    console.print("\n" + "=" * 79, style="bold blue")
    console.print("📈 Stock Market Analysis Tool   ", style="bold magenta", justify="center")
    console.print("=" * 79, style="bold blue")
    console.print("\n[bold yellow]Welcome to the Stock Market Analysis Tool![/bold yellow]")
    console.print(
        "[cyan]Analyze historical stock data, calculate indicators, make predictions, and export results.[/cyan]\n")

    # Get user input for the stock ticker and date range
    ticker = input("Enter the stock ticker symbol (e.g., AAPL): ").strip().upper()
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default: ").strip()
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for default: ").strip()

    # Fetch stock data
    data = fetch_stock_data(ticker, start_date, end_date)
    if data.empty:
        console.print("[bold red]No data available. Exiting program.[/bold red]")
        return

    # Calculate indicators
    data = calculate_indicators(data)

    # Show stock insights
    provide_insights(data)

    # Predict future stock prices
    predictions = predict_stock_prices(data)

    # Ask user if they want to export the data
    export_choice = input("Would you like to export the data? (yes/no): ").strip().lower()
    if export_choice == "yes":
        export_data(data, ticker)

    # Launch the dashboard
    console.print("\n[bold green]Launching interactive dashboard...[/bold green]")
    create_dashboard(data, ticker, predictions)


# Run the program
if __name__ == "__main__":
    main()










