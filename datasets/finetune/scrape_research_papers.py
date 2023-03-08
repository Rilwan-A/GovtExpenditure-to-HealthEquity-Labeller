from __future__ import annotations

import os
import sys
sys.path.append(os.getcwd())

from io import BytesIO

from argparse import ArgumentParser

import yaml
import multiprocessing as mp
import itertools

import random

import aiohttp
from aiohttp import ClientConnectorError, ClientConnectorSSLError, TCPConnector

import asyncio

import logging

from typing import List, Sequence
from bs4 import BeautifulSoup
import requests

import fitz
import csv 
import wget

import gzip as gz
from fake_headers import Headers
from requests_html import HTMLSession, AsyncHTMLSession

import ssl
import certifi



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
    source:str,
    mp_count=4):

    #TODO: add tqdm to the multiproc maps

    search_terms = yaml.safe_load(open('./datasets/finetune/search_terms.yaml','r'))

    # scrape pdfs
    logging.info("Scraping pdfs")
    # li_li_pdf_title_authors = scrape_pdfs( search_terms, downloads_per_search_term, min_citations ) 
    li_li_pdf_title_authors = asyncio.run(scrape_pdfs( search_terms, downloads_per_search_term, min_citations, source ) )
    logging.info("Finished Scraping pdfs")

    # save pdfs to file
    logging.info("Saving pdfs to file")
    with mp.Pool(mp_count) as p:
        res = p.starmap(save_pdfs,  list(zip(search_terms, itertools.count(), li_li_pdf_title_authors))  )
    
    # extract texts
    logging.info("Covnerting pdf to txt")
    li_pdfs = [pdf for pdf, title, author in sum(li_li_pdf_title_authors, []) ] # all pdfs flattened
    with mp.Pool(mp_count) as p:
        gen_texts = p.imap(extract_text, li_pdfs, chunksize=1  )

        # Replacing pdfs in li_li_pdf_title_authors with processed text
        li_li_txt_title_author = li_li_pdf_title_authors

        for idx_searchterm in range(len(li_li_pdf_title_authors)):
            
            for idx_rsearchpaper in range(len(li_li_pdf_title_authors[idx_searchterm])):
                
                new_vals = ( next( gen_texts ), 
                             *li_li_txt_title_author[idx_searchterm][idx_rsearchpaper][1:]
                             )

                li_li_txt_title_author[idx_searchterm][idx_rsearchpaper] =  new_vals
            
    
    # Saving Texts to file
    logging.info("Saving txt to file")
    with mp.Pool(mp_count) as p:
        res = p.starmap(save_text, list(zip(search_terms, itertools.count(), li_li_txt_title_author)) )
    
    logging.info("Script Finished")
    return None

    
async def scrape_pdfs( search_terms, downloads_per_search_term, min_citations, source:str ) -> List[ List[tuple[str,bytes]]  ]:
    
    async with aiohttp.ClientSession(headers={'User-Agent':'Mozilla/5.0' } ) as session:
    # ,connector=TCPConnector(ssl=False) ) as session:
    # async with aiohttp.ClientSession() as session:
        logging.info("Started aiohttp Session")
        

        li_tasks = [None]*len(search_terms)
        
        if source == 'google_scholar':
            scrape_func =  scrape_pdfs_google_scholar
        elif source == 'semantic_scholar':
            scrape_func = get_pdfs_semantic_scholar_api

        for idx, search_term in enumerate(search_terms):
            # li_tasks[idx] = asyncio.create_task(scrape_pdfs_google_scholar(session, search_term, downloads_per_search_term, min_citations))
            
            li_tasks[idx] = scrape_func(session, search_term, downloads_per_search_term, min_citations)
                    
        li_pdf_title_author = await asyncio.gather(*li_tasks)

    return li_pdf_title_author

async def scrape_pdfs_google_scholar(session, search_term:str, downloads_per_search_term:int, min_citations:int) -> List[tuple[str, bytes]]:
    
    # Helper class to generate headers for requests
    # NOTE: TO avoid brotli encoded responses, ".update({'accept-encoding': 'gzip, deflate, utf-8'})":" is appended to the generate output
    headers = Headers(os='win', headers=True)

    docs_per_url = 10
    
    li_pdf = []
    li_title = []
    li_author = [] #NOTE: currently author not scraped

    # open webpage
    for idx in itertools.count():
    
        url = f"https://scholar.google.com/scholar?start={docs_per_url*idx}&q={search_term.replace(' ','+')}&hl=en"

        async with session.get(url, headers=headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'}) ) as resp:
            # if no more pages then break
            if resp.status != 200:
                break

            # convert to beautilful soup tree
            text = await resp.read()
            soup = BeautifulSoup(text, 'lxml')

        # searching and extracting pdf links

        ## getting the html divisions representing a single research paper
        tags_research_papers = soup.find_all( 'div', {'class','gs_r gs_or gs_scl'}, recursive=True )

        ## filtering for html div tag (for research papers) that have a link to a pdf file
        for idx1 in reversed(range(len(tags_research_papers))):
            
            tag = tags_research_papers[idx1]
            res = tag.find(href= lambda txt: (txt is not None) and txt[-4:]=='.pdf' )

            if res is None:
                tags_research_papers.pop(idx1)

        ## filtering for html div representations of research papers that have at least min_citations citations
        for idx2 in reversed(range(len(tags_research_papers))):
            tag = tags_research_papers[idx2]

            # research paper has a child tag with text 'Cited by N' where N at least min citations
            res = tag.find( string = lambda txt:  ('Cited by' in txt) and (min_citations<=int( txt.split(' ')[-1]) ) )

            if res is None:
                tags_research_papers.pop(idx2)

        ### titles are the text in <a> tags that have a parent tag with class name 'gs_rt'
        titles = [ ''.join(tag.find( class_='gs_rt' ).strings) for tag in tags_research_papers]   

        ## extracting pdf urls
        urls = [ tag.find(href= lambda txt: (txt is not None) and txt[-4:]=='.pdf' ).attrs['href'] for tag in tags_research_papers ]
        
        # extracting pdf document
        # ca_certs = certifi.where()
        # ssl_context = ssl.create_default_context(cafile=ca_certs)
        # ssl_context.set_ciphers("DEFAULT")

        

        # pdfs = [ await (await session.get(url, cookies=resp.cookies,
        #                  headers=headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'}),
        #                 #  verify_ssl=False,
        #                 #  ssl=False,
        #                     ssl_context = ssl_context
        #                  )).content.read()
        #             for url in urls]
        pdfs = []
        for url in urls:
            try:
                pdf = await (await session.get(url, cookies=resp.cookies,
                            headers=headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'}),
                            #  verify_ssl=False,
                            #  ssl=False,
                                # ssl_context = ssl_context
                            )).content.read()
            except (ClientConnectorSSLError, ClientConnectorError) as e:
                pdf = "NAN"
            pdfs.append(pdf)


        # Note: Cloudfare blocking disables all pdfs linked via ResearchGate, remove downloads that were content blocked
        _ =[(pdf,title) for pdf,title in zip(pdfs,titles) if pdf[:4]==b'%PDF']
        if len(_)>0:
            pdfs,titles =  list( zip(*_))
        else:
            pdfs,titles = [],[]

        li_title.extend(titles)
        li_pdf.extend(pdfs)
        
        # break pdf urls collected exceeds min citations
        await asyncio.sleep(5.5)
        if len(li_pdf)>=downloads_per_search_term:
            break
            
    # filter urls on existence of download link, and min citations, link containing text [PDF]
    li_pdf = li_pdf[:downloads_per_search_term]
    li_title = li_title[:downloads_per_search_term]
    li_author = ['']*len(li_pdf)

    outp = [ (pdf, title, author) for pdf, title, author in zip(li_pdf, li_title, li_author) ]
    return outp


async def get_pdfs_semantic_scholar_api(session, search_term:str, downloads_per_search_term:int, min_citations:int) -> List[tuple[str, bytes]]:
    
    # Helper class to generate headers for requests
    # NOTE: TO avoid brotli encoded responses, ".update({'accept-encoding': 'gzip, deflate, utf-8'})":" is appended to the generate output
    headers = Headers(os='win', headers=True)
    
    li_pdf = []
    li_title = []
    li_author = []

    papers_per_query = 100
    # open webpage
    for idx in itertools.count(start=0):
            
        url = \
        "https://api.semanticscholar.org/graph/v1/paper/search?query=covid&year=2020-2023&openAccessPdf&fieldsOfStudy=Physics,Philosophy&fields=title,year,authors"
        
        url_base = "https://api.semanticscholar.org/graph/v1/paper/search?"
        url_query = f"query={search_term.replace(' ','+')}"
        url_filters = "openAccessPdf"
        url_fields = "fields=title,authors,citationCount"
        url_paper_count = f"offset={str(idx*papers_per_query)}&limit={str(papers_per_query)}"
        
        url = url_base+'&'.join([url_query, url_filters, url_fields, url_paper_count])

        headersList = {
            "Accept": "*/*",
            "Content-Type": "application/json" 
            }
        async with session.get(url, headers=headersList, timeout=120 ) as resp:
    
            if resp.status != 200:
                break

            resp_dict = await resp.content.read()

            # if no more pages then break
            if resp_dict['total'] < idx*papers_per_query: break


        li_dict_papers = resp_dict['data']


        ## filtering for html div representations of research papers that have at least min_citations citations
        for idx in reversed(range(len(li_dict_papers))):
            dict_paper = li_dict_papers[idx]

            if dict_paper['citationCount'] < min_citations:
                li_dict_papers.pop(idx)
            
        # extracting pdf documents        
        pdfs = [ await (await session.get(d['openAccessPdf']['url'], cookies=resp.cookies,
                         headers=headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'})  )).content.read()
                    for d in li_dict_papers]
      

        # Filtering out invalid pdfs
        titles, authors, pdfs  = zip( *[ (pdf, d['title'], d['authors'] ) for d, pdf in zip(li_dict_papers,pdfs) if pdf.content[:4]==b'%PDF'  ]  )
        
        li_pdf.extend(pdfs)
        li_title.extend(titles)
        li_author.extend(authors)

        # break pdf urls collected exceeds min citations
        if len(li_pdf)>=downloads_per_search_term:
            li_pdf = li_pdf[:downloads_per_search_term]
            li_title = li_title[:downloads_per_search_term]
            li_author = li_author[:downloads_per_search_term]
            break
        else:
            await asyncio.sleep(2.0)

    # s = requests.session()
    # res = s.get('https://discovery.ucl.ac.uk/10106434/3/Bockenhauer_BMJ%20Ten%20years%20essay2pg3.pdf', headers=headers.generate().update({'accept-encoding': 'gzip, deflate, utf-8'}))

    outp = [ (pdf, title, author) for pdf, title, author in zip(li_pdf, li_title, li_author) ]
    return outp

def save_pdfs(search_term, search_term_idx, li_pdf_title_author):
    
    # making directory
    dir_ = f'./datasets/finetune/pdf_format/{search_term_idx:02}'
    os.makedirs(dir_, exist_ok=True)

    with open( os.path.join(dir_,'search_term.txt'), 'w') as f:
        f.write(search_term)

    # Saving index mapping file_numbers to paper titles
    fp_index = os.path.join(dir_, 'index.csv')
    with open(fp_index, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(  [ (idx, title, author) for idx, (pdf, title, author) in enumerate(li_pdf_title_author) ] ) 

    # Saving documents
    for idx, (pdf, title, author) in enumerate(li_pdf_title_author):
        fp_file = os.path.join(dir_, f"{idx:03}.pdf")
        with open(fp_file, "wb") as f:
            f.write( pdf )
    
    return None

def extract_text(pdf:bytes) -> str:
    
    #TODO akanni.ade : ignore first page
    #TODO akanni.ade : ignore contents page if it exists
    #TODO akanni.ade : ensure that text on images is ignored
    #TODO akanni.ade : ensure reference section is ignored
    #TODO akanni.ade : pages that couldn't be parsed
    #TODO akanni.ade : ensure text is sectioned by pages that couldn't be parsed
    #TODO akanni.ade : ensure appendix section is dropped

    doc = fitz.Document( stream=pdf )
    text = ''
    for page in doc:
        text += page.get_text()

    return text

def save_text(search_term:str, search_term_idx:int, li_txt_title_author: List[List[str,str,str]]):

    # making directory
    dir_ = f'./datasets/finetune/text_format/{search_term_idx:02}'
    os.makedirs(dir_, exist_ok=True)

    with open( os.path.join(dir_,'search_term.txt'), 'w') as f:
        f.write(search_term)

    # Saving index mapping file_numbers to paper titles
    fp_index = os.path.join(dir_, 'index.csv')
    with open(fp_index, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(  [ (idx, title, author) for idx, (txt, title, author) in enumerate(li_txt_title_author) ] ) 


    for idx, (txt, title, author) in enumerate(li_txt_title_author):
        fp_file = os.path.join(dir_, f"{idx:03}.txt.gz")
        with gz.open(fp_file, "wb") as f:
            f.write( txt.encode('utf-8') )
              

def parse_args(parent_parser):
    if parent_parser != None:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, allow_abbrev=False)
    else:
        parser = ArgumentParser()

    parser.add_argument('--downloads_per_search_term', default=5, type=int, help='Number of documents to download per search term')
    parser.add_argument('--min_citations', type=int, default=0, help='Minimum number of citations for a paper to have to be included in download')
    parser.add_argument('--mp_count', type=int, default=1, help='')
    parser.add_argument('--source', type=str, default='semantic_scholar', help='Which website to use for sourcing the research papers',choices=['google_scholar','semantic_scholar'])
    args = parser.parse_known_args()[0]

    return args

if __name__ == '__main__':

    parser = ArgumentParser(add_help=False, allow_abbrev = False)
    
    args = parse_args(parser)

    main(**vars(args))