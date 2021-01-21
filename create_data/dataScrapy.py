#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import json
import os
import time


def resolve_content_book(barcode, save_to_dir):

    r = requests.get('http://iiif.onb.ac.at/presentation/ABO/+%s/manifest/' %barcode)
    manifest = json.loads(r.text)
    # store the meta data
    with open(os.path.join(save_to_dir, '%s.meta' %barcode), 'w') as metadata:
        try:
            json.dump(manifest['metadata'], fp=metadata, sort_keys=True, indent=4)
        except:
            print('failed to load metadata')
    # store the content
    with open(os.path.join(save_to_dir,'%s.txt' %barcode), 'w') as fulltext:
        for canvas in manifest['sequences'][0]['canvases']:
            content = canvas['otherContent'][0]['resources'][0]['resource']['@id']
            if content[-3:] == 'txt':
                canvas_text = requests.get(content)
                status = canvas_text.status_code
                if status == 200:
                    pass
                else:
                    print('Http error: %d!' %status)
                fulltext.write(canvas_text.text)
                # save all pages to a single .txt file, use ########## to denote.
                fulltext.write('\n##########\n')
            time.sleep(0.01)


def getContentForManifest(barcode, save_to_dir):
    # if alredy resolved content, then fetch file directly
    directory = os.path.join(save_to_dir, '{}.txt'.format(barcode))
    if os.path.isfile(directory):
        print('book already downloaded.')
    else:
        resolve_content_book(barcode, save_to_dir)

    with open(directory, 'r') as fulltext:
        fullcontent = fulltext.read()
    return fullcontent


if __name__ == '__main__':
    barcode = 'Z114805705'
    book = getContentForManifest(barcode)