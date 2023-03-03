from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd())

from argparse import ArgumentParser

import yaml
import multiprocessing as mp
import itertools

import aiohttp
import asyncio

import logging

from typing import List, Sequence
from bs4 import BeautifulSoup
import requests

import pymupdf
import fitz

"""
    This script scrapes research papers from Google Scholar.
    Then converts the research papers to a .txt document format.
    Then performs preprocessing.
    Then saves in pkl format.

    Fastest pdf to text extractor is PyMuPDF (benchmark experiments here https://github.com/py-pdf/benchmarks)
"""

def main(
    downloads_per_search_term:int,
    min_citations:int,
    debugging:bool=False):

    search_terms = yaml.safe_load('.datasets/finetune/search_terms.yaml')

    # scrape pdfs / htmls
    scraped_pdfs = scrape_pdfs( search_terms, downloads_per_search_term, min_citations )
    
    # save pdfs to file
    for search_term, scraped_pdfs in zip():

    # convert pdfs / htmls to text files or hdf4 files
    map_idx_search_term = { idx:search_term for idx,search_term in enumerate(search_terms) }

    text_docs = preprocess_pdf( map_idx_search_term[idx], 
                                idx,
                                scrape_pdfs )

    


async def scrape_pdfs( search_terms, downloads_per_search_term, min_citations ):

    
    async with aiohttp.ClientSession() as session:
        logging.info("Started aiohttp Session")
        
        li_tasks = [None]*len(search_terms)
        
        for idx, search_term in enumerate(search_terms):
            li_tasks[idx] = asyncio.create_task(scrape_docs_for_search_term(session, search_term, downloads_per_search_term, min_citations))
        
    li_pdfs = [await task for task in li_tasks]

    return li_pdfs

async def scrape_docs_for_search_term(session, search_term:str, downloads_per_search_term:int, min_citations:int) -> List[tuple[str, bytes]]
    
    docs_per_url = 10
    
    li_docs_url = []
    li_docs_titles = []

    # open webpage
    for idx in itertools.count():
    
        url = f"https://scholar.google.com/scholar?start={docs_per_url*idx}&q={search_term.replace(' ','+')}&hl=en"

        async with session.get(url) as resp:
            # convert to beautilful soup tree
            text = await resp.read()
            soup = BeautifulSoup(text, 'lxml')

            # searching and extracting pdf links

            ## getting the html divisions representing a single research paper
            tags_research_papers = soup.find_all( lambda tag: getattr(tag, 'class', None)== 'gs_r gs_or gs_scl' )
            # tags_research_papers = soup.find_all( class='gs_r gs_or gs_scl')

            ## filtering for html div representations of research papers that have a link to a pdf file
            for idx in reversed(range(len(tags_research_papers))):
                tag = tags_research_papers[idx]

                res = tag.find( lambda sub_tag: hasattr(sub_tag,'href') == True and getattr(sub_tag, 'href')[-4:]=='.pdf' )

                if len(res)==0:
                    tags_research_papers.pop(idx)

            ## filtering for html div representations of research papers that have at least min_citations citations
            for idx in reversed(range(len(tags_research_papers))):
                tag = tags_research_papers[idx]

                # research paper has a child tag with text 'Cited by N'
                res = tag.find( lambda sub_tag: 'Cited by' in sub_tag.string and min_citations<=int(sub_tag.string.split(' ')[-1]) )

                if len(res)==0:
                    tags_research_papers.pop(idx)

            ## extracting pdf urls
            li_urls = [ tag.find(lambda sub_tag: getattr(sub_tag, 'href')[-4:]=='.pdf').href for tag in tags_research_papers ]
            ### titles are the text in <a> tags that have a parent tag with class name 'gs_rt'
            li_titles = [ tag.find( lambda sub_tag: sub_tag.name == 'a' and getattr(sub_tag.parent,'class') == 'gs_rt').text ]
            # removing <b> tags from text


            li_docs_url.extend(li_urls)
            li_docs_titles.extend(li_titles)

        # break pdf urls collected exceeds min citations
        if len(li_docs_url)>=downloads_per_search_term:
            break

    # filter urls on existence of download link, and min citations, link containing text [PDF]
    li_docs_url = li_docs_url[:downloads_per_search_term]

    
    li_pdfs = [ requests.get(url) for url in li_docs_url]
    
    return li_pdfs # in pdf format

def save_pdfs(search_term, search_term_idx, pdfs):
    
    # making directory
    dir_ = f'.datasets/finetune/datasets/{search_term_idx:02}'
    os.makedirs(dir_, exist_ok=True)

    for pdf_title, pdf_name in pdfs:



def convert_to_strings():
    pass

def parse_args(parent_parser):
    if parent_parser != None:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
    else:
        parser = ArgumentParser()

    parser.add_argument('--downloads_per_search_term', default=50, type=int, help='Number of documents to download per search term')
    parser.add_argument('--min_citations', type=int, default=50, help='Minimum number of citations for a paper to have to be included in download')
    parser.add_argument('--debugging', action='store_true')
    args = parser.parse_known_args()[0]

    return args

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False, allow_abbrev = False)
    
    args = parse_args(parser)

    main(*args)