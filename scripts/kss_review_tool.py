"""Offline Windows WAV listening review; only explicit reviewer decisions are saved."""
import argparse
import csv
from datetime import datetime,timezone,timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import wave


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


class ReviewStore:
    def __init__(self,bundle,priority,output):
        self.bundle=Path(bundle).resolve();self.output=Path(output).resolve()
        self.source=self.bundle/'spec_input.json';self.source_hash=digest(self.source)
        payload=json.loads(self.source.read_text(encoding='utf-8'))
        self.original={r['utterance_id']:r for r in payload['utterances']}
        with Path(priority).open(encoding='utf-8-sig',newline='') as stream:
            self.rows=list(csv.DictReader(stream))
        if len({r['utterance_id'] for r in self.rows})!=len(self.rows) or set(self.original)!={r['utterance_id'] for r in self.rows}:
            raise ValueError('Priority list does not match the source IDs')
        for row in self.rows:
            original=self.original[row['utterance_id']]
            if row['sha256']!=original['original_sha256'] or row['reference']!=original['text']:
                raise ValueError('Priority transcript/hash mismatch')
        self.output.mkdir(parents=True,exist_ok=True)
        metadata=self.output/'source.json'
        expected={'source_manifest_sha256':self.source_hash,'priority_sha256':digest(priority)}
        if metadata.exists() and json.loads(metadata.read_text())!=expected:
            raise ValueError('Review workspace belongs to different source data')
        if not metadata.exists():
            metadata.write_text(json.dumps(expected,indent=2),encoding='utf-8')
        self.journal=self.output/'review_events.jsonl';self.decisions={}
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                record=json.loads(line)
                if record['utterance_id'] not in self.original or record['audio_sha256']!=self.original[record['utterance_id']]['original_sha256']:
                    raise ValueError('Review history source mismatch')
                self.decisions[record['utterance_id']]=record
        self.playback=None

    def audio(self,uid):
        path=(self.bundle/self.original[uid]['original_path']).resolve()
        if self.bundle not in path.parents or digest(path)!=self.original[uid]['original_sha256']:
            raise ValueError('Original audio path/hash mismatch')
        return path

    def begin_playback(self,uid):
        path=self.audio(uid)
        with wave.open(str(path),'rb') as audio:
            seconds=audio.getnframes()/audio.getframerate()
        self.playback=(uid,time.monotonic(),seconds)
        return path

    def decide(self,uid,decision,reviewer,memo):
        if decision not in ('예','아니오','보류'):
            raise ValueError('Unknown decision')
        elapsed=0.
        if decision!='보류':
            if not self.playback or self.playback[0]!=uid:
                raise ValueError('먼저 이 발화를 재생해 주세요.')
            elapsed=time.monotonic()-self.playback[1]
            if elapsed<self.playback[2]:
                raise ValueError('끝까지 듣고 판정을 저장해 주세요.')
        if not reviewer.strip():
            raise ValueError('검수자 ID를 입력해 주세요. 실명은 필요하지 않습니다.')
        self.audio(uid)
        record={'utterance_id':uid,'heard_and_matches(예/아니오)':decision if decision!='보류' else '',
            'review_status':'MATCH_CONFIRMED_BY_REVIEWER' if decision=='예' else 'MISMATCH_REPORTED' if decision=='아니오' else 'PENDING',
            'checked_by':reviewer.strip(),'date':datetime.now(timezone(timedelta(hours=9))).isoformat(),
            'memo':memo.strip(),'audio_sha256':self.original[uid]['original_sha256'],
            'playback_elapsed_seconds':round(elapsed,3)}
        if digest(self.source)!=self.source_hash:
            raise ValueError('Source manifest changed; stop review')
        with self.journal.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        self.decisions[uid]=record
        self.export()

    def export(self):
        destination=self.output/'transcript_check_reviewed.csv'
        temporary=self.output/'transcript_check_reviewed.csv.partial'
        fields=['utterance_id','wav_path','transcript','heard_and_matches(예/아니오)','checked_by','date','memo','audio_sha256','review_status']
        with temporary.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in self.rows:
                uid=row['utterance_id'];original=self.original[uid];saved=self.decisions.get(uid,{})
                result={k:saved.get(k,'') for k in fields}
                result.update(utterance_id=uid,wav_path=str(self.audio(uid)),transcript=original['text'],
                              audio_sha256=original['original_sha256'],review_status=saved.get('review_status','UNREVIEWED'))
                writer.writerow(result)
        os.replace(temporary,destination)


def build_ui(root,store):
    import tkinter as tk
    from tkinter import ttk,messagebox
    root.title('SafeSpeech KSS 청취 검수 — 허가 상태는 변경하지 않습니다')
    root.geometry('1160x780')
    reviewer=tk.StringVar(value='reviewer_01');position=tk.IntVar(value=0);heading=tk.StringVar()
    status=tk.StringVar(value='F5 재생 · F8 일치 · F9 불일치 · F10 보류. ASR 일치는 검수가 아닙니다.')
    frame=ttk.Frame(root,padding=12);frame.pack(fill='both',expand=True)
    ttk.Label(frame,text='검수자 ID (실명 불필요)').pack(anchor='w');ttk.Entry(frame,textvariable=reviewer).pack(anchor='w')
    ttk.Label(frame,textvariable=heading,font=('맑은 고딕',12,'bold')).pack(anchor='w',pady=8)
    body=ttk.Frame(frame);body.pack(fill='both',expand=True)
    listing=ttk.Treeview(body,columns=('id','cer','state'),show='headings',height=17)
    for key,label,width in [('id','발화',150),('cer','ASR CER',85),('state','검수',130)]:
        listing.heading(key,text=label);listing.column(key,width=width)
    listing.pack(side='left',fill='y')
    right=ttk.Frame(body,padding=(15,0));right.pack(side='left',fill='both',expand=True)
    ttk.Label(right,text='원래 대본').pack(anchor='w');reference=tk.Text(right,height=4,wrap='word',font=('맑은 고딕',13));reference.pack(fill='x')
    ttk.Label(right,text='ASR 전사 (검수 보조 정보)').pack(anchor='w',pady=(12,0));hypothesis=tk.Text(right,height=4,wrap='word');hypothesis.pack(fill='x')
    ttk.Label(right,text='메모 / 수정 제안 (원본 대본을 직접 수정하지 않음)').pack(anchor='w',pady=(12,0));memo=tk.Text(right,height=5,wrap='word');memo.pack(fill='x')
    ttk.Label(right,text='원본 검수·사용 허가 플래그는 자동 변경하지 않습니다.\n판정은 별도 CSV와 변경 이력에 저장됩니다.',foreground='#994411').pack(anchor='w',pady=12)
    buttons=ttk.Frame(frame);buttons.pack(fill='x',pady=10)

    def refresh():
        for item in listing.get_children():listing.delete(item)
        for i,row in enumerate(store.rows):
            saved=store.decisions.get(row['utterance_id'],{})
            listing.insert('', 'end',iid=str(i),values=(row['utterance_id'],f"{float(row['normalized_cer']):.3f}",saved.get('heard_and_matches(예/아니오)','미검수') or '보류'))

    def show(index):
        if not 0<=index<len(store.rows):return
        if os.name=='nt':
            import winsound
            winsound.PlaySound(None,0)
        store.playback=None;position.set(index);row=store.rows[index];uid=row['utterance_id']
        count=sum(r.get('heard_and_matches(예/아니오)')=='예' for r in store.decisions.values())
        heading.set(f"{index+1}/{len(store.rows)} · {uid} · {row['split']} · 일치 확인 {count}개")
        for widget,text in [(reference,row['reference']),(hypothesis,row['asr_hypothesis'])]:
            widget.configure(state='normal');widget.delete('1.0','end');widget.insert('1.0',text);widget.configure(state='disabled')
        memo.delete('1.0','end');memo.insert('1.0',store.decisions.get(uid,{}).get('memo',''))
        listing.selection_set(str(index));listing.see(str(index))

    def play():
        try:
            import winsound
            uid=store.rows[position.get()]['utterance_id'];path=store.begin_playback(uid)
            winsound.PlaySound(str(path),winsound.SND_FILENAME|winsound.SND_ASYNC)
            status.set('재생 중입니다. 끝까지 듣고 직접 판정을 선택해 주세요.')
        except Exception as error:
            store.playback=None;messagebox.showerror('재생 오류',str(error))

    def decide(value):
        try:
            uid=store.rows[position.get()]['utterance_id'];store.decide(uid,value,reviewer.get(),memo.get('1.0','end'))
            refresh();show(min(position.get()+1,len(store.rows)-1));status.set(f'{uid}: {value} 저장. 허가 플래그는 그대로입니다.')
        except Exception as error:messagebox.showerror('저장 전 확인',str(error))

    def select(event):
        selected=listing.selection()
        if selected and int(selected[0])!=position.get():show(int(selected[0]))

    for text,action in [('이전',lambda:show(position.get()-1)),('재생 F5',play),('일치 F8',lambda:decide('예')),
        ('불일치 F9',lambda:decide('아니오')),('보류 F10',lambda:decide('보류')),('다음',lambda:show(position.get()+1)),
        ('기록 폴더',lambda:os.startfile(store.output))]:
        ttk.Button(buttons,text=text,command=action).pack(side='left',padx=3)
    ttk.Label(frame,textvariable=status,wraplength=1100).pack(anchor='w')
    listing.bind('<<TreeviewSelect>>',select)
    for key,action in [('<F5>',play),('<F8>',lambda:decide('예')),('<F9>',lambda:decide('아니오')),('<F10>',lambda:decide('보류'))]:
        root.bind(key,lambda event,call=action:call())
    refresh();show(0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--bundle',type=Path,default=root/'runs/desktop_20261118/data/data_set/kss_part_200')
    p.add_argument('--priority',type=Path,default=root/'runs/kss_asr_screening_20gow51e/listening_priority.csv')
    p.add_argument('--output',type=Path,default=root/'runs/kss_listening_review_200')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    import tkinter as tk
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            store=ReviewStore(args.bundle,args.priority,Path(directory))
            window=tk.Tk();window.withdraw();build_ui(window,store);window.update_idletasks();window.destroy()
            if store.decisions:raise RuntimeError('Self-test unexpectedly wrote decisions')
            print('UI construction PASS; no audio playback or human review performed')
    else:
        store=ReviewStore(args.bundle,args.priority,args.output)
        window=tk.Tk();build_ui(window,store);window.mainloop()


if __name__=='__main__':main()
