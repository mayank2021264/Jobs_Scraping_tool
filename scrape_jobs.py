import streamlit as st
from jobspy import scrape_jobs
import pandas as pd
from datetime import datetime
from pathlib import Path
import csv
import logging
from typing import List, Optional, Dict

# Copy your JobScraper class here (unchanged)
class JobScraper:
    """A class to handle job scraping operations with better configuration and error handling."""
    
    # Essential columns that must be present in output
    REQUIRED_COLUMNS = {
        'title': '',           # Job title
        'company': '',         # Company name
        'location': '',        # Job location
        'salary': 'Not specified',  # Salary information
        'job_url': '',         # Link to the job posting
        'description': '',     # Job description
        'posted_at': 'Not specified',  # When the job was posted
        'schedule_type': 'Not specified',  # Full-time/Part-time/Contract
        'site': ''            # Source website (LinkedIn, Indeed, etc.)
    }
    
    def __init__(
        self,
        search_term: str,
        location: str,
        output_dir: str = "job_results",
        site_names: List[str] = None,
        results_wanted: int = 20,
        hours_old: int = 72,
        country: str = "INDIA",
        linkedin_fetch_description: bool = False,
        proxies: List[str] = None
    ):
        self.search_term = search_term
        self.location = location
        self.output_dir = Path(output_dir)
        self.site_names = site_names or ["indeed", "linkedin", "google"]
        self.results_wanted = results_wanted
        self.hours_old = hours_old
        self.country = country
        self.linkedin_fetch_description = linkedin_fetch_description
        self.proxies = proxies
        
        self._setup_logging()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Configure logging for the scraper."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'job_scraper_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _prepare_google_search_term(self) -> str:
        """Prepare the Google-specific search term."""
        return f"{self.search_term} jobs near {self.location} since yesterday"

    def _save_to_file(self, jobs_df: pd.DataFrame, file_format: str = 'csv') -> str:
        """Save the jobs data to a file in the specified format."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jobs_{timestamp}"
        
        try:
            if file_format.lower() == 'csv':
                output_path = self.output_dir / f"{filename}.csv"
                jobs_df.to_csv(
                    output_path,
                    quoting=csv.QUOTE_NONNUMERIC,
                    escapechar="\\",
                    index=False
                )
            elif file_format.lower() == 'excel':
                output_path = self.output_dir / f"{filename}.xlsx"
                jobs_df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
                
            self.logger.info(f"Successfully saved jobs data to {output_path}")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Error saving file: {str(e)}")
            raise

    def _clean_data(self, jobs_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the scraped job data."""
        if jobs_df.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS.keys())
        
        # Create a new DataFrame with only the required columns
        cleaned_df = pd.DataFrame(columns=self.REQUIRED_COLUMNS.keys())
        
        # Copy existing data
        for col in self.REQUIRED_COLUMNS.keys():
            if col in jobs_df.columns:
                cleaned_df[col] = jobs_df[col]
            else:
                cleaned_df[col] = self.REQUIRED_COLUMNS[col]
        
        # Remove duplicates based on job title and company
        cleaned_df = cleaned_df.drop_duplicates(subset=['title', 'company'], keep='first')
        
        # Fill missing values with default values
        for col, default_value in self.REQUIRED_COLUMNS.items():
            cleaned_df[col] = cleaned_df[col].fillna(default_value)
        
        # Try to extract schedule type from description if not available
        if 'description' in cleaned_df.columns:
            def extract_schedule_type(description):
                description = str(description).lower()
                if 'full-time' in description or 'full time' in description:
                    return 'Full-time'
                elif 'part-time' in description or 'part time' in description:
                    return 'Part-time'
                elif 'contract' in description:
                    return 'Contract'
                else:
                    return 'Full-time'  # Default to Full-time if not specified
            
            # Only update schedule_type if it's not already specified
            mask = cleaned_df['schedule_type'] == 'Not specified'
            cleaned_df.loc[mask, 'schedule_type'] = cleaned_df.loc[mask, 'description'].apply(extract_schedule_type)
        
        # Try to extract salary information from description if not available
        def extract_salary(row):
            if row['salary'] != 'Not specified':
                return row['salary']
            
            description = str(row['description']).lower()
            # Add basic salary pattern matching here if needed
            return 'Not specified'
        
        cleaned_df['salary'] = cleaned_df.apply(extract_salary, axis=1)
        
        # Ensure columns are in the correct order
        cleaned_df = cleaned_df[list(self.REQUIRED_COLUMNS.keys())]
        
        return cleaned_df

    def scrape(self, save_format: str = 'csv') -> Dict:
        """
        Main method to scrape jobs and save results.
        Returns a dictionary with the results summary.
        """
        self.logger.info(f"Starting job scrape for '{self.search_term}' in {self.location}")

        all_jobs = []

        try:
            for site in self.site_names:
                try:
                    jobs = scrape_jobs(
                        site_name=site,
                        search_term=self.search_term,
                        location=self.location,
                        results_wanted=self.results_wanted,
                        hours_old=self.hours_old,
                        country_indeed=self.country,
                        google_search_term=self._prepare_google_search_term() if site == "google" else None,
                        linkedin_fetch_description=self.linkedin_fetch_description,
                        proxies=self.proxies
                    )

                    if isinstance(jobs, pd.DataFrame):
                        all_jobs.append(jobs)
                    else:
                        all_jobs.append(pd.DataFrame(jobs))

                    self.logger.info(f"JobSpy:{site.capitalize()} - finished scraping")
                except Exception as e:
                    self.logger.error(f"JobSpy:{site.capitalize()} - error during scraping: {str(e)}")

            if not all_jobs:
                raise ValueError("No job data scraped from any source.")

            jobs_df = pd.concat(all_jobs, ignore_index=True)
            jobs_df = self._clean_data(jobs_df)
            output_path = self._save_to_file(jobs_df, save_format)

            results = {
                "total_jobs": len(jobs_df),
                "unique_companies": jobs_df['company'].nunique() if not jobs_df.empty else 0,
                "output_path": output_path,
                "timestamp": datetime.now().isoformat()
            }

            self.logger.info(f"Scraping completed. Found {results['total_jobs']} jobs")
            return results

        except Exception as e:
            self.logger.error(f"Error during scraping: {str(e)}")
            raise

# Streamlit App
def main():
    st.set_page_config(
        page_title="Job Scraper",
        page_icon="🔍",
        layout="centered"
    )
    
    # Custom CSS for cards
    st.markdown("""
    <style>
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            margin-bottom: 20px;
            background-color: white;
        }
        .card-title {
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #1E88E5;
        }
        .stButton>button {
            width: 100%;
            padding: 10px;
            border-radius: 5px;
            background-color: #1E88E5;
            color: white;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header Section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🔍 Online Job Scraper</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 16px;'>Find your dream job by scraping multiple sources with custom filters</p>", unsafe_allow_html=True)
    
    # Main Content Container
    with st.container():
        # Search Parameters Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Search Parameters</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            search_term = st.text_input("Job Title", "software engineer")
            location = st.text_input("Location", "Delhi, India")
            results_wanted = st.number_input("Number of Results", min_value=5, max_value=100, value=20)
            
        with col2:
            hours_old = st.number_input("Max Age (hours)", min_value=1, max_value=720, value=72)
            country = st.selectbox("Country", ["INDIA", "USA", "UK", "CANADA", "AUSTRALIA"], index=0)
            linkedin_fetch_description = st.checkbox("Fetch LinkedIn Descriptions", value=False)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Sources Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Select Job Sources</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            indeed = st.checkbox("Indeed", value=True)
        with col2:
            linkedin = st.checkbox("LinkedIn", value=True)
        with col3:
            google = st.checkbox("Google", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Output Options Card
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Output Options</div>", unsafe_allow_html=True)
        file_format = st.radio("Select output format:", ["CSV", "Excel"], index=0, horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Scrape Button (centered)
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("✨ Scrape Jobs Now", type="primary"):
                with st.spinner("Scraping job postings from selected sources..."):
                    try:
                        # Prepare site names based on checkboxes
                        site_names = []
                        if indeed: site_names.append("indeed")
                        if linkedin: site_names.append("linkedin")
                        if google: site_names.append("google")
                        
                        if not site_names:
                            st.error("Please select at least one job source!")
                            st.stop()
                        
                        scraper = JobScraper(
                            search_term=search_term,
                            location=location,
                            results_wanted=results_wanted,
                            hours_old=hours_old,
                            country=country,
                            site_names=site_names,
                            linkedin_fetch_description=linkedin_fetch_description
                        )
                        
                        results = scraper.scrape(save_format=file_format.lower())
                        
                        # Results Card
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<div class='card-title'>✨ Scraping Results</div>", unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Jobs Found", results['total_jobs'])
                        with col2:
                            st.metric("Unique Companies", results['unique_companies'])
                        
                        # Display sample data
                        st.markdown("**Sample Job Postings:**")
                        jobs_df = pd.read_csv(results['output_path']) if file_format == "CSV" else pd.read_excel(results['output_path'])
                        st.dataframe(jobs_df.head(5))
                        
                        # Download button
                        with open(results['output_path'], "rb") as file:
                            btn = st.download_button(
                                label=f"📥 Download Full Results ({file_format})",
                                data=file,
                                file_name=f"jobs_{datetime.now().strftime('%Y%m%d')}.{file_format.lower()}",
                                mime="text/csv" if file_format == "CSV" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"An error occurred during scraping: {str(e)}")

if __name__ == "__main__":
    main()