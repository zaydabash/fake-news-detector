# Fake News Detector

A machine learning system for detecting fake news articles using traditional NLP techniques. This project implements a robust pipeline for training, evaluating, and deploying fake news detection models with production-ready code quality.

## Problem Statement

In today's digital age, misinformation and fake news spread rapidly across social media and news platforms. This tool addresses the critical need for automated fake news detection by providing a reliable, interpretable system that can classify news articles as real or fake based on their textual content.

## Key Features

- **Robust Data Pipeline**: Automated download and preprocessing of the LIAR dataset
- **Feature Engineering**: TF-IDF vectorization with configurable parameters
- **Multiple Models**: Logistic Regression and Linear SVM with automatic model selection
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and feature analysis
- **Production Ready**: CLI interface and Streamlit web application
- **Reproducible**: Seeded runs and version-controlled artifacts

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Make (optional, for convenience commands)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd fake-news-detector

# Set up environment and install dependencies
make setup

# Activate virtual environment (if not already active)
source venv/bin/activate
```

### Running the Multi-Niche Pipeline

```bash
# Train models for specific niches
make data NICHE=climate_claims
make train NICHE=climate_claims
make evaluate NICHE=climate_claims

make data NICHE=celebrity_rumors
make train NICHE=celebrity_rumors

# Train unified model across all niches
make unified-train
make unified-eval

# Legacy LIAR dataset (backwards compatibility)
make data-liar
make train-liar
make evaluate-liar
```

### Making Predictions

```bash
# Single text prediction (uses best available model)
make predict TEXT="Your news text here"

# Interactive mode
python -m src.predict.cli interactive

# Batch prediction from CSV
python -m src.predict.cli predict-file data.csv --text-column text
```

### Web Interface

```bash
# Launch Streamlit demo
make demo
# or
streamlit run app/streamlit_app.py
```

## Multi-Niche Data Cards

This system supports multiple specialized niches for targeted misinformation detection. Each niche uses domain-specific sources and weak labeling heuristics.

### 🌍 Climate Claims Detection

**Niche**: `climate_claims`  
**Description**: Distinguishes between climate hype/denial claims and scientific climate reporting.

**Data Sources**:
- **Claim Sources**: wattsupwiththat.com, climatedepot.com, iceagenow.info, climatechangereconsidered.org
- **Scientific Sources**: ipcc.ch, noaa.gov, climate.nasa.gov, nature.com, carbonbrief.org

**Weak Labeling Rules**:
- **Claims**: Contains ≥2 of {"hoax", "climate scam", "ice age next year", "cities underwater", "climate lockdown"}
- **Scientific**: Contains {"IPCC", "radiative forcing", "anomaly", "confidence interval", "peer-reviewed", "methodology"}

**Known Limitations**:
- Domain bias: Heavy reliance on source reputation
- Temporal bias: Climate language evolves rapidly
- Complexity bias: May miss nuanced scientific discussions

### 👑 Celebrity Rumors Detection

**Niche**: `celebrity_rumors`  
**Description**: Separates entertainment rumors from verified celebrity reporting.

**Data Sources**:
- **Rumor Sources**: tmz.com, pagesix.com, radaronline.com, the-sun.com, perezhilton.com
- **Verified Sources**: apnews.com, variety.com, hollywoodreporter.com, rollingstone.com, people.com

**Weak Labeling Rules**:
- **Rumors**: Contains ≥2 of {"allegedly", "insider says", "spotted with", "leak", "confirmed?"} AND lacks official source
- **Verified**: Mentions {"press release", "official statement", "representative said", "confirmed by"}

**Known Limitations**:
- Source hierarchy bias: May favor established outlets over emerging media
- Context dependency: Official vs. unofficial sources can be ambiguous
- Temporal sensitivity: Rumors may later become verified

### 💰 Hustle Scams Detection

**Niche**: `hustle_scams`  
**Description**: Identifies "get rich quick" scams vs. legitimate financial guidance.

**Data Sources**:
- **Scam Sources**: hustlersuniversity.com, getrichquicktips.com, crypto-fastcash.com, secretmethod.biz
- **Legitimate Sources**: irs.gov, ftc.gov, sec.gov, investor.gov, bloomberg.com, wsj.com

**Weak Labeling Rules**:
- **Scams**: Contains ≥2 of {"$10k/day", "zero risk", "secret method", "DM to join", "limited spots", "financial freedom in 30 days"}
- **Legitimate**: Contains {"risk disclosure", "fiduciary", "regulator", "SEC filing", "Form 10-K", "advisory"}

**Known Limitations**:
- Regulatory bias: May miss legitimate but aggressive marketing
- Language evolution: Scam tactics adapt to avoid detection
- Cultural context: Financial advice varies by region and culture

### 🤖 AI Hype Detection

**Niche**: `ai_hype`  
**Description**: Distinguishes AI hype from sober technical reporting.

**Data Sources**:
- **Hype Sources**: futurism.com, singularityhub.com, ai-news.com, techcrunch.com/ai
- **Technical Sources**: nature.com, arxiv.org, openai.com, deepmind.com, mit.edu

**Weak Labeling Rules**:
- **Hype**: Contains {"breakthrough", "revolutionary", "game-changing", "world-changing", "unprecedented"}
- **Technical**: Contains {"study shows", "research indicates", "peer-reviewed", "methodology", "limitations"}

### 🏫 Campus Rumor Detection

**Niche**: `campus_rumor`  
**Description**: Separates campus rumors from official university communications.

**Data Sources**:
- **Rumor Sources**: collegeconfidential.com, reddit.com/r/college, campusgossip.com
- **Official Sources**: harvard.edu, mit.edu, stanford.edu, berkeley.edu, princeton.edu

**Weak Labeling Rules**:
- **Rumors**: Contains {"heard from", "rumor has it", "word is", "apparently", "supposedly"}
- **Official**: Contains {"official announcement", "university statement", "confirmed by", "press release"}

### 🏥 Medical Claims Detection

**Niche**: `medical_claims`  
**Description**: Identifies medical misinformation vs. evidence-based health reporting.

**Data Sources**:
- **Misinformation Sources**: naturalnews.com, mercola.com, greenmedinfo.com, thetruthaboutcancer.com
- **Evidence-Based Sources**: nih.gov, cdc.gov, who.int, nejm.org, thelancet.com, mayoclinic.org

**Weak Labeling Rules**:
- **Misinformation**: Contains {"big pharma", "cover-up", "conspiracy", "natural cure", "miracle treatment"}
- **Evidence-Based**: Contains {"clinical trial", "peer-reviewed study", "randomized controlled trial", "evidence-based"}

### 💎 Crypto Scams Detection

**Niche**: `crypto_scams`  
**Description**: Detects cryptocurrency scams vs. legitimate crypto information.

**Data Sources**:
- **Scam Sources**: cryptopump.com, mooncoin.com, getrichcrypto.com, pumpanddump.crypto
- **Legitimate Sources**: coinbase.com, binance.com, coinmarketcap.com, coindesk.com, sec.gov

**Weak Labeling Rules**:
- **Scams**: Contains {"guaranteed returns", "100x gains", "moon soon", "pump and dump", "get rich quick"}
- **Legitimate**: Contains {"investment risk", "volatility", "regulatory compliance", "due diligence"}

### 📊 Original LIAR Dataset

**Dataset**: LIAR (Liar, Liar Pants on Fire)  
**Source**: [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

**Description**: The LIAR dataset contains 12,836 short statements labeled for truthfulness by PolitiFact.

**Label Mapping**:
- **Real News** (0): `true`, `mostly-true`, `half-true`
- **Fake News** (1): `barely-true`, `false`, `pants-fire`

**Statistics**:
- Total samples: 12,836
- Training set: 10,269 samples
- Validation set: 1,284 samples  
- Test set: 1,283 samples
- Class distribution: ~52% Real, ~48% Fake

## Model Architecture

### Feature Engineering
- **Text Preprocessing**: Lowercase, URL/email removal, tokenization
- **TF-IDF Vectorization**: 
  - Max features: 10,000
  - N-gram range: (1, 2)
  - Min document frequency: 3
  - Stop word removal: English

### Model Selection
- **Logistic Regression**: Linear baseline with class balancing
- **Linear SVM**: Strong linear classifier with class balancing
- **Selection Criteria**: Cross-validation F1-score (macro average)

## Results

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.623 | 0.621 | 0.623 | 0.622 | 0.687 |
| Linear SVM | 0.619 | 0.617 | 0.619 | 0.618 | 0.684 |

*Results on LIAR test set with binary classification*

### Key Insights

- **Feature Analysis**: Real news tends to use more formal language and specific facts, while fake news often contains emotional language and vague claims
- **Top Real News Indicators**: Government, policy, official, report, federal
- **Top Fake News Indicators**: Claim, false, misleading, conspiracy, secret

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── Makefile
├── .pre-commit-config.yaml
├── src/
│   ├── data/           # Data download and preprocessing
│   ├── features/       # Feature engineering (TF-IDF)
│   ├── models/         # Model training and evaluation
│   ├── predict/        # Prediction CLI
│   └── utils/          # Utility functions
├── data/
│   ├── raw/           # Raw LIAR dataset
│   ├── interim/       # Processed datasets
│   └── processed/     # Final train/val/test splits
├── artifacts/
│   ├── models/        # Trained models
│   ├── vectorizers/   # Feature extractors
│   └── reports/       # Evaluation reports and plots
├── app/               # Streamlit web application
├── tests/             # Unit tests
└── examples/          # Example data and scripts
```

## Configuration

### Model Parameters

Key parameters can be adjusted in the respective modules:

- **TF-IDF**: `max_features`, `ngram_range`, `min_df` in `src/features/tfidf.py`
- **Logistic Regression**: `C`, `solver`, `max_iter` in `src/models/train.py`
- **SVM**: `C`, `max_iter` in `src/models/train.py`

### Data Processing

- **Text cleaning**: Modify `clean_text()` in `src/data/clean.py`
- **Tokenization**: Adjust `tokenize_text()` parameters
- **Class balancing**: Automatic via `compute_class_weight`

## Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_features.py -v

# Run with coverage
python -m pytest tests/ --cov=src
```

## Limitations and Ethics

### Limitations

1. **Domain Specificity**: Model trained on political statements may not generalize to other domains
2. **Language Bias**: English-only training data limits applicability to other languages
3. **Temporal Drift**: Training data from 2007-2016 may not reflect current misinformation patterns
4. **Context Dependency**: Short statements lack full context that human fact-checkers use
5. **Adversarial Robustness**: Model may be vulnerable to adversarial text modifications

### Ethical Considerations

1. **Bias Awareness**: Model may perpetuate biases present in training data
2. **Misuse Prevention**: Tool should not be used as sole source for content moderation
3. **Transparency**: All predictions should include confidence scores and uncertainty estimates
4. **Human Oversight**: Automated systems should complement, not replace, human fact-checking

### Responsible Use

- Use as a preliminary screening tool, not final arbiter
- Always verify critical decisions with human review
- Consider cultural and linguistic context when applying to new domains
- Regularly evaluate model performance and retrain with new data

## Future Improvements

### Planned Enhancements

1. **DistilBERT Integration**: Fine-tuned transformer model for improved accuracy
2. **Multi-modal Analysis**: Incorporate metadata (source, author, date)
3. **Real-time Updates**: Continuous learning from new fact-checking data
4. **Multi-language Support**: Extend to other languages and cultural contexts
5. **Adversarial Training**: Improve robustness against adversarial examples

### Branch Structure

- `main`: Current stable version with traditional ML
- `distilbert`: Experimental transformer-based approach
- `multimodal`: Extended features including metadata
- `production`: Optimized deployment configuration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{fake_news_detector,
  title={Fake News Detector: A Machine Learning Approach},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/fake-news-detector}
}
```

## Acknowledgments

- LIAR dataset creators for providing the benchmark data
- PolitiFact for fact-checking annotations
- Scikit-learn, NLTK, and other open-source libraries
- Streamlit for the web interface framework
